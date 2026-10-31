"""0단계: YAMNet을 '점수 + 임베딩' 두 출력의 TFLite로 내보낸다(기기 개인화 힌트용). TF 2.15에서 실행.

    python sim/export_yamnet_tflite.py    # → artifacts/yamnet_emb.tflite (약 15MB, 표준 연산만 사용)

TF Hub의 yamnet.tflite는 점수(521)만 내보내서, 아기별 개인화에 필요한 임베딩(1024)이 없다.
입력은 기존과 같은 15600샘플(0.975초 @16kHz)이고, 출력은 embedding(1024)과 scores(521) 두 개다.
변환 뒤 TF Hub 원본과 임베딩·점수가 같은지 확인한다.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "artifacts", "yamnet_emb.tflite")


def main():
    import tensorflow as tf
    import tensorflow_hub as hub
    m = hub.load("https://tfhub.dev/google/yamnet/1")

    @tf.function(input_signature=[tf.TensorSpec([15600], tf.float32)])
    def f(x):
        scores, emb, _ = m(x)
        return {"scores": scores[0], "embedding": emb[0]}

    conv = tf.lite.TFLiteConverter.from_concrete_functions([f.get_concrete_function()], m)
    tfl = conv.convert()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "wb").write(tfl)

    it = tf.lite.Interpreter(model_content=open(OUT, "rb").read())   # 한글 경로는 model_path로 못 연다
    it.allocate_tensors()
    rng = np.random.default_rng(0)
    worst = 1.0
    for _ in range(5):
        x = (rng.standard_normal(15600) * 0.1).astype(np.float32)
        it.set_tensor(it.get_input_details()[0]["index"], x)
        it.invoke()
        out = {int(d["shape"][-1]): it.get_tensor(d["index"]).reshape(-1) for d in it.get_output_details()}
        rs, re_, _ = m(x)
        e, r = out[1024], re_.numpy()[0]
        worst = min(worst, float(e @ r / (np.linalg.norm(e) * np.linalg.norm(r))))
        assert np.allclose(out[521], rs.numpy()[0], atol=1e-3)
    print(f"저장: {OUT} ({len(tfl) / 1e6:.1f}MB), 출력 {sorted(out)}, 임베딩 최소 코사인 {worst:.5f}")


if __name__ == "__main__":
    main()
