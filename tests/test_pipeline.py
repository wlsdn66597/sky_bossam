"""Synthetic fixtures test plumbing only; these are not infant performance results."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as Args

import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from camera_fall.common import dump,load,rows,sha256,write_rows
from camera_fall.data import register,split_data,prepare_videos
from camera_fall.detector import Detector
from camera_fall.evaluation import tune,evaluate
from camera_fall.temporal import train
from camera_fall.video import replay,extract


class PipelineTests(unittest.TestCase):
    def test_onnx_video_features_r0_and_r1(self):
        import cv2
        import onnx
        from onnx import helper,TensorProto,numpy_helper
        old=os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                Path("fixture").mkdir()
                # A constant output graph checks adapter compatibility, not model accuracy.
                raw=np.array([[[208.],[208.],[80.],[100.],[.9]]],dtype=np.float32)
                graph=helper.make_graph([helper.make_node("Constant",[],["output"],value=numpy_helper.from_array(raw))],"synthetic_constant_fixture",[helper.make_tensor_value_info("images",TensorProto.FLOAT,[1,3,416,416])],[helper.make_tensor_value_info("output",TensorProto.FLOAT,[1,5,1])])
                model=helper.make_model(graph,opset_imports=[helper.make_opsetid("",17)])
                model.ir_version=9
                weights=Path("fixture/body.onnx")
                onnx.save(model,weights)
                dump("fixture/model_meta.json",dict(classes=["baby_body"],imgsz=416,sha256=sha256(weights),parity_passed=True,fixture_only=True))
                detector=Detector(weights)
                boxes=detector.predict(np.zeros((480,640,3),np.uint8))
                self.assertEqual(len(boxes),1)
                geometry=dict(camera_setup_id="fixture_camera",coordinate_system="normalized_image_xy",reference_width=64,reference_height=48,calibration_valid=True,bed_polygon=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],boundaries=[dict(id="right",type="open",points=[[.9,.1],[.9,.9]])])
                dump("data/geometry/fixture_camera.json",geometry)
                policy=dict(schema_version=1,policy="geometry_temporal_v1",near_distance_ratio=.1,outward_speed_ratio_per_s=.02,warning_hold_s=.6,clear_hold_s=1.,max_observation_gap_s=.5,alarm_merge_gap_s=2.)
                dump("policy.json",policy)
                videos,events=[],[]
                for i in range(3):
                    file=Path(f"fixture/video{i}.mp4")
                    writer=cv2.VideoWriter(str(file),cv2.VideoWriter_fourcc(*"mp4v"),10,(64,48))
                    self.assertTrue(writer.isOpened())
                    for j in range(200):
                        writer.write(np.full((48,64,3),10+j,np.uint8))
                    writer.release()
                    videos.append(dict(video_id=f"v{i}",dataset_id="fixture",path=str(file),source_group=f"g{i}",session_id=f"s{i}",event_group_id=f"e{i}",subject_id=f"synthetic{i}",subject_kind="synthetic",camera_setup_id="fixture_camera",modality="rgb",width=64,height=48,fps_nominal=10,duration_s=20.,annotations_reviewed=True,annotation_coverage=[[0,20]]))
                    events.append(dict(event_id=f"exit{i}",video_id=f"v{i}",event_group_id=f"e{i}",event_type="unassisted_bed_exit",action_start_s=10.,critical_ts_s=15.,end_s=17.,surrogate=True,visibility="clear",boundary_id="right"))
                Path("terms.txt").write_text("Synthetic test fixture only.")
                register(Args(input="fixture",dataset="fixture",terms_file="terms.txt",approve_for_training=True,version="1",source_url="synthetic:test",registry="data/registry/datasets.json"))
                write_rows("data/manifests/videos.jsonl",videos)
                write_rows("data/manifests/events.jsonl",events)
                prepare_videos(Args(videos="data/manifests/videos.jsonl",events="data/manifests/events.jsonl",registry="data/registry/datasets.json",extract=False))
                split_data(Args(mode="grouped",duplicate_review=None,hash_distance=4,seed=42,out="data/splits"))
                replay(Args(out="reports/replay",geometry="data/geometry/fixture_camera.json",policy="policy.json",weights=str(weights),imgsz=416,device="cpu",confidence=.25,source="fixture/video0.mp4",sample_hz=5.,timestamps_file=None,video_id="v0",overlay=True))
                self.assertEqual(load("reports/replay/runtime_benchmark.json")["processed_frames"],100)
                for split in ("train","val","test"):
                    extract(Args(policy="policy.json",split=split,weights=str(weights),imgsz=416,device="cpu",confidence=.25,sample_hz=5.))
                cache=str(Path("data/features")/sha256(weights))
                tune(Args(split="val",features=cache,events="data/manifests/events.jsonl",config="policy.json",out="reports/selected_r0.yaml"))
                evaluate(Args(policy="r0",split="test",features=cache,events="data/manifests/events.jsonl",frozen_config="reports/selected_r0.yaml",out="reports/event_metrics.json"))
                metrics=load("reports/event_metrics.json")["summary"]
                self.assertEqual(metrics["target_events"],1)
                self.assertEqual(metrics["early_event_recall"],0.)
                config=dict(model="causal_gru",sample_hz=5,history_steps=16,input_dim=40,hidden_size=8,num_layers=1,dropout=0.,horizons_s=[1,3,5],batch_size=16,epochs=2,patience=1,lr=.001,weight_decay=.0001,seed=42,max_gap_s=.5,max_train_windows_per_video=1000)
                dump("temporal.json",config)
                train(Args(config="temporal.json",out="runs/r1",features=cache,events="data/manifests/events.jsonl",model="r1",expected_gpu="L4"))
                exported=load("runs/r1/r1_coefficients.json")
                self.assertEqual(len(exported["models"]),3)
                self.assertTrue(Path("runs/r1/validation_horizons.json").is_file())
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()
