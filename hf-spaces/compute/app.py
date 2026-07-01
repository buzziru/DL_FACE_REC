"""HF compute space — face age/gender/emotion over ONNX.

Called by the static demo (ingyoun/face-rec-demo) via @gradio/client on upload,
and usable directly. Detection: RetinaFace (uniface ONNX). Classifiers: the
project's own age/gender/emotion models converted to ONNX.
"""
import gradio as gr
from inference import FacePipeline, draw

pipe = FacePipeline()


def analyze(image):
    if image is None:
        return None, []
    results = pipe.predict(image)
    annotated = draw(image, results)
    faces = [
        {
            "idx": i + 1,
            "age": r["age"],
            "gender": r["gender"],
            "female_prob": r["female_prob"],
            "emotion": r["emotion"],
        }
        for i, r in enumerate(results)
    ]
    return annotated, faces


with gr.Blocks(title="Face Age·Gender·Emotion") as demo:
    gr.Markdown(
        "## 얼굴 나이·성별·감정 추정 (ONNX)\n"
        "RetinaFace 검출 후 얼굴마다 나이(회귀)·성별·감정(6클래스)을 동시 추정합니다."
    )
    with gr.Row():
        inp = gr.Image(type="numpy", label="입력 이미지")
        out_img = gr.Image(type="numpy", label="결과")
    out_json = gr.JSON(label="faces")
    btn = gr.Button("분석", variant="primary")
    btn.click(analyze, inputs=inp, outputs=[out_img, out_json], api_name="predict")
    inp.upload(analyze, inputs=inp, outputs=[out_img, out_json])

if __name__ == "__main__":
    demo.launch()
