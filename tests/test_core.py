from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from camera_fall.common import dump,load,rows,write_rows
from camera_fall.data import components,convert_labels,register,prepare,split_data,validate_events
from camera_fall.detector import decode,letterbox,nms,audit_training_data
from camera_fall.evaluation import aggregate,alarm_episodes,match_events
from camera_fall.geometry import geometry_features,validate_geometry
from camera_fall.risk import FeatureExtractor,R0,FEATURE_ORDER
from camera_fall.temporal import fit_preprocess,targets_at,transform,windows


G = dict(camera_setup_id="test",coordinate_system="normalized_image_xy",reference_width=640,reference_height=480,calibration_valid=True,bed_polygon=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],boundaries=[dict(id="right",type="open",points=[[.9,.1],[.9,.9]])])
C = dict(schema_version=1,policy="geometry_temporal_v1",near_distance_ratio=.15,outward_speed_ratio_per_s=.01,warning_hold_s=.4,clear_hold_s=.4,max_observation_gap_s=.5,alarm_merge_gap_s=2.)


def det(x):
    return [dict(bbox=[x-.04,.4,x+.04,.6],confidence=.9)]


def feature(t,distance=.05,speed=.05,valid=True,track=1):
    values = {k:None for k in FEATURE_ORDER}
    values.update(det_valid=int(valid),geometry_valid=1,open_edge_distance=distance,approach_speed=speed)
    return dict(video_ts_s=t,track_id=track,boundary_id="right" if valid else None,observed=valid,values=[values[k] for k in FEATURE_ORDER],masks=[int(values[k] is not None) for k in FEATURE_ORDER])


class LabelTests(unittest.TestCase):
    def test_baby_crib_classes_merge(self):
        names={0:"baby",1:"baby_prone",2:"baby_sideways",3:"baby_supine"}
        out,original=convert_labels("3 .5 .5 .2 .2\n1 .3 .3 .1 .1",names)
        self.assertEqual(len(out),2)
        self.assertTrue(all(x.startswith("0 ") for x in out))
        self.assertEqual(original,["baby_supine","baby_prone"])

    def test_face_not_body(self):
        out,_=convert_labels("0 .5 .5 .2 .2",{0:"Airway-Covered",1:"Baby-Prone"})
        self.assertEqual(out,[])

    def test_bad_labels_rejected(self):
        for text in ("0 .5 .5 -1 .2","0 nan .5 .2 .2","0 .99 .5 .2 .2","7 .5 .5 .2 .2","0 .5 .5 .2 .2 .1"):
            with self.subTest(text=text),self.assertRaises(ValueError):
                convert_labels(text,{0:"baby"})
        with self.assertRaises(ValueError):
            convert_labels("",{0:"person"})


class SplitTests(unittest.TestCase):
    def test_transitive_source_and_duplicate_groups(self):
        records=[dict(image_id="a",source_group="s1",sha256="x"),dict(image_id="b",source_group="s1",sha256="y"),dict(image_id="c",source_group="s2",sha256="y"),dict(image_id="d",source_group="s3",sha256="z")]
        groups=components(records)
        self.assertIn([0,1,2],groups)
        self.assertIn([3],groups)

    def test_same_video_and_multicamera_event(self):
        records=[dict(image_id="a",video_id="v1"),dict(video_id="v1",event_group_id="same"),dict(video_id="v2",event_group_id="same")]
        self.assertEqual(components(records),[[0,1,2]])


class GeometryTests(unittest.TestCase):
    def test_overlap(self):
        f=geometry_features([.8,.4,1.,.6],G)
        self.assertAlmostEqual(f["outside_ratio"],.5)
        self.assertIsNone(f["rail_edge_distance"])

    def test_signed_distance(self):
        self.assertGreater(geometry_features([.75,.4,.85,.6],G)["open_edge_distance"],0)
        self.assertLess(geometry_features([.92,.4,.98,.6],G)["open_edge_distance"],0)

    def test_other_edge_exit_does_not_reverse_open_right_distance(self):
        f=geometry_features([.01,.4,.07,.6],G)
        self.assertGreater(f["open_edge_distance"],.5)

    def test_bad_calibration(self):
        for polygon in ([[.1,.1],[.9,.9],[.1,.9],[.9,.1]],[[.1,.1],[.5,.4],[.9,.1],[.9,.9],[.1,.9]]):
            with self.assertRaises(ValueError):
                validate_geometry(dict(G,bed_polygon=polygon))


class TrackingTests(unittest.TestCase):
    def test_speed_uses_past_only(self):
        ex=FeatureExtractor(G)
        a=ex.step(0,det(.7),.5)
        frozen=copy.deepcopy(a)
        b=ex.step(.2,det(.72),.5)
        self.assertEqual(a,frozen)
        self.assertGreater(b["values"][FEATURE_ORDER.index("approach_speed")],0)
        self.assertIsNone(a["values"][FEATURE_ORDER.index("approach_speed")])

    def test_missing_and_replacement_reset_velocity(self):
        ex=FeatureExtractor(G)
        a=ex.step(0,det(.3),.5)
        ex.step(.2,[],.5)
        b=ex.step(.4,det(.32),.5)
        self.assertIsNone(b["values"][8])
        c=ex.step(.6,det(.8),.5)
        self.assertNotEqual(a["track_id"],c["track_id"])
        self.assertIsNone(c["values"][8])

    def test_multiple_candidates_not_guessed(self):
        ex=FeatureExtractor(G)
        r=ex.step(0,det(.3)+det(.8),.5)
        self.assertFalse(r["observed"])


class PolicyTests(unittest.TestCase):
    def test_warning_dwell_and_missing_are_not_clear(self):
        r=R0(C)
        self.assertFalse(r.step(feature(0))["active_warning"])
        self.assertFalse(r.step(feature(.2))["active_warning"])
        result=r.step(feature(.4))
        self.assertEqual(result["events"][0]["event_type"],"risk_started")
        result=r.step(feature(.6,valid=False))
        self.assertEqual(result["state"],"unavailable")
        self.assertTrue(result["active_warning"])
        self.assertEqual(result["events"],[])

    def test_gap_cannot_accumulate_dwell(self):
        r=R0(C)
        r.step(feature(0))
        self.assertFalse(r.step(feature(2))["active_warning"])

    def test_rail_proximity_never_open_edge_warning(self):
        r=R0(C)
        for t in (0,.2,.4,.6):
            f=feature(t,distance=None)
            f["values"][7]=.02
            result=r.step(f)
            self.assertEqual(result["state"],"edge_approach")

    def test_track_replacement_restarts_hold(self):
        r=R0(C)
        r.step(feature(0));r.step(feature(.2))
        self.assertFalse(r.step(feature(.4,track=2))["active_warning"])

    def test_stationary_near_boundary_does_not_clear_active_warning(self):
        r=R0(C)
        for t in (0,.2,.4): r.step(feature(t))
        for t in (.6,.8,1.): self.assertTrue(r.step(feature(t,speed=0))["active_warning"])


class EvaluationTests(unittest.TestCase):
    def test_late_and_duplicate_not_early_tp(self):
        events=[dict(event_id="a",action_start_s=2.,critical_ts_s=5.,end_s=7.)]
        result=match_events(events,[1.,5.1,5.2])
        self.assertEqual(len(result["early"]),0)
        self.assertEqual(len(result["late"]),1)
        self.assertEqual(result["false_alarms"],2)

    def test_one_alarm_cannot_match_two_events(self):
        events=[dict(event_id="a",action_start_s=2.,critical_ts_s=5.,end_s=7.),dict(event_id="b",action_start_s=3.,critical_ts_s=6.,end_s=8.)]
        self.assertEqual(len(match_events(events,[4.])["early"]),1)

    def test_alarm_merge_uses_episode_end(self):
        events=[dict(event_type="risk_started",video_ts_s=1),dict(event_type="risk_ended",video_ts_s=10),dict(event_type="risk_started",video_ts_s=11)]
        self.assertEqual(alarm_episodes(events,20,2),[[1,20]])

    def test_no_positive_is_na_not_perfect(self):
        result=aggregate([dict(targets=0,alarms=0,early=[],late=[],false_alarms=0,duration_s=10,valid_s=5)])
        self.assertIsNone(result["early_event_recall"])
        self.assertEqual(result["availability"],.5)


class TransformTests(unittest.TestCase):
    def test_letterbox_restore_non_square(self):
        tensor,transform=letterbox(np.zeros((480,640,3),np.uint8),416)
        self.assertEqual(tensor.shape,(1,3,416,416))
        # Original box x=[160,480], y=[120,360] -> scale .65, top=52.
        raw=np.array([[[208],[208],[208],[156],[.9]]],dtype=np.float32)
        out=decode(raw,transform)
        np.testing.assert_allclose(out[0]["bbox"],[.25,.25,.75,.75],atol=1e-6)

    def test_legacy_parser_rejected(self):
        with self.assertRaises(ValueError):
            decode(np.zeros((1,9,100)),(1,0,0,640,480))

    def test_nms(self):
        boxes=np.array([[0,0,1,1],[0,0,1,1],[2,2,3,3]],dtype=float)
        self.assertEqual(nms(boxes,np.array([.9,.8,.7])),[0,2])


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.v=dict(video_id="v",duration_s=10,annotations_reviewed=True,annotation_coverage=[[0,10]])
        self.e=[dict(video_id="v",event_type="unassisted_bed_exit",critical_ts_s=8,action_start_s=6,end_s=9)]

    def test_incomplete_future_is_unknown(self):
        self.assertEqual(targets_at(9,[1,3,5],self.v,[])[1],[1.,0.,0.])

    def test_after_exit_excluded(self):
        self.assertEqual(targets_at(8,[1,3,5],self.v,self.e)[1],[0.,0.,0.])

    def test_positive_target_only_future(self):
        y,m=targets_at(6,[1,3],self.v,self.e)
        self.assertEqual(y,[0.,1.]);self.assertEqual(m,[1.,1.])

    def test_missing_is_zero_only_after_scaling(self):
        a=feature(0);b=feature(.2)
        a["values"][0]=.2;a["masks"][0]=1
        b["values"][0]=.8;b["masks"][0]=1
        prep=fit_preprocess([(self.v,[a,b])])
        c=feature(.4)
        z=transform([c],prep)
        self.assertEqual(z[0,0],0)
        self.assertEqual(z[0,20],0)
        self.assertEqual(prep["mean"][15],0)


class DataIntegrationTest(unittest.TestCase):
    def test_register_convert_original_split_and_tamper_guard(self):
        from PIL import Image
        old=os.getcwd()
        with tempfile.TemporaryDirectory() as temp:
            os.chdir(temp)
            try:
                raw=Path("raw");raw.mkdir()
                dump(raw/"data.yaml",dict(names=["baby","baby_prone","baby_sideways","baby_supine"]))
                for i,split in enumerate(("train","valid","test")):
                    (raw/split/"images").mkdir(parents=True)
                    (raw/split/"labels").mkdir()
                    Image.new("RGB",(32,32),(20+i*50,30,40)).save(raw/split/"images"/"one.png")
                    (raw/split/"labels"/"one.txt").write_text("3 .5 .5 .5 .5\n")
                Path("terms.txt").write_text("Synthetic test fixture. No infant data.")
                register(SimpleNamespace(input="raw",dataset="fixture",terms_file="terms.txt",approve_for_training=True,version="1",source_url="synthetic:test",registry="data/registry/datasets.json"))
                prepare(SimpleNamespace(registry="data/registry/datasets.json",dataset="fixture",groups=None))
                split_data(SimpleNamespace(mode="original",duplicate_review=None,hash_distance=4,seed=42,out="data/splits"))
                self.assertEqual(audit_training_data(True)["image_counts"],dict(train=1,val=1,test=1))
                with self.assertRaises(ValueError): audit_training_data(False)
                label=next(Path("data/detection/train/labels").glob("*.txt"))
                label.write_text("")
                with self.assertRaises(ValueError): audit_training_data(True)
            finally:
                os.chdir(old)

    def test_video_pts_are_monotonic(self):
        import cv2
        from camera_fall.video import frames
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"fixture.mp4"
            writer=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*"mp4v"),10,(64,48))
            self.assertTrue(writer.isOpened())
            for i in range(20): writer.write(np.full((48,64,3),i,np.uint8))
            writer.release()
            samples=list(frames(p,sample_hz=5))
            self.assertEqual(len(samples),10)
            self.assertAlmostEqual(samples[-1][1],1.8,places=3)


if __name__ == "__main__":
    unittest.main()
