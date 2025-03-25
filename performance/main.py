import time
import cv2
import typer
import asyncio

from typing import Callable, List
from pathlib import Path
from tqdm import tqdm
from furiosa.runtime import create_queue
from warboy.yolo.preprocess import PreProcessor
from warboy.yolo.postprocess import ObjDetPostprocess

app = typer.Typer(
    help=f"Warboy Test CLI", add_completion=False
)

def decorate_with_bar(string: str) -> str:
    """Decorate given string with bar

    Args:
        string: String to decorate

    Returns:
        Decorated string
    """

    bar = "----------------------------------------------------------------------"
    return "\n".join([bar, string, bar])


def time_with_proper_suffix(t: float, digits: int = 5) -> str:
    """Returns time with proper suffix

    Args:
        t: Time in seconds
        digits: Number of digits after decimal point

    Returns:
        Time with proper suffix
    """

    units = iter(["sec", "ms", "us", "ns"])
    while t < 1:
        t *= 1_000
        next(units)
    return f"{t:.{digits}f} {next(units)}"


def decorate_result(
    total_time: float,
    queries: int,
    header: str = "",
    digits: int = 5,
    newline: bool = True,
) -> str:
    """Decorate benchmark result

    Args:
        total_time: Total elapsed time
        queries: Number of queries
        header: Header string
        digits: Number of digits after decimal point
        newline: Whether to add newline at the end

    Returns:
        Decorated benchmark result
    """

    result = []
    result.append(decorate_with_bar(header))
    result.append(f"Total elapsed time: {time_with_proper_suffix(total_time, digits)}")
    result.append(f"QPS: {queries / total_time:.{digits}f}")
    result.append(
        f"Avg. elapsed time / sample: {time_with_proper_suffix(total_time / queries, digits)}"
    )
    if newline:
        result.append("")
    return "\n".join(result)

async def submit_with(submitter, model_inputs):
    idx = 0
    for model_input, context in tqdm(model_inputs, desc="Inference"):
        await submitter.submit(model_input, context=(context, idx, ))
        idx += 1

async def recv_with(receiver, model_outputs, input_imgs):
    while True:
        try:
            async def recv():
                context, outputs = await receiver.recv()
                return context, outputs

            task = asyncio.create_task(recv())
            (context, idx), outputs = await asyncio.wait_for(task, timeout=0.1)
            model_outputs.append((outputs, context, input_imgs[idx]))
        except asyncio.TimeoutError:  
            break

    return model_outputs

async def warboy_inference(submitter, receiver, model_inputs, model_outputs, input_imgs):
    model_ouputs = []
    submit_task = asyncio.create_task(submit_with(submitter, model_inputs))
    recv_task = asyncio.create_task(recv_with(receiver, model_outputs, input_imgs))
    await submit_task
    outputs = await recv_task
    return outputs

async def run_inference(
    model: str,
    input_paths: str,
    preprocess: Callable,
    postprocess: Callable,
    device_str="warboy(2)*1",
):
    warning = """WARN: the benchmark results may depend on the number of input samples,sizes of the images, and a machine where this benchmark is running."""
    queries = len(input_paths)
    print(f"Run benchmark on {queries} input samples ...")
    print(decorate_with_bar(warning))

    async with create_queue(model = model, device=device_str, worker_num = 16) as (submitter, receiver):
        model_inputs, model_outputs, input_imgs = [], [], []
        
        initial_time = time.perf_counter()
        for input_path in tqdm(input_paths, desc="Preprocess"):
            img = cv2.imread(input_path)
            input_imgs.append(img)
            model_inputs.append(preprocess(img))

        after_preprocess = time.perf_counter()
        outputs = await warboy_inference(submitter, receiver, model_inputs, model_outputs, input_imgs)
        after_npu = time.perf_counter()

        for output, context, input_img in tqdm(
            outputs, desc="Postprocess"
        ):
            postprocess(output, context, input_img)
        all_done = time.perf_counter()

    print(
        decorate_result(
            all_done - initial_time, queries, "Preprocess -> Inference -> Postprocess"
        )
    )
    print(
        decorate_result(
            all_done - after_preprocess, queries, "Inference -> Postprocess"
        )
    )
    print(
        decorate_result(
            after_npu - after_preprocess, queries, "Inference", newline=False
        )
    )


def resolve_input_paths(input_path: Path) -> List[str]:
    """Create input file list"""
    if input_path.is_file():
        return [str(input_path)]
    elif input_path.is_dir():
        # Directory may contain image files
        image_extensions = {".jpg", ".jpeg", ".png"}
        return [
            str(p.resolve())
            for p in input_path.glob("**/*")
            if p.suffix.lower() in image_extensions
        ]
    else:
        typer.echo(f"Invalid input path '{str(input_path)}'")
        raise typer.Exit(1)

@app.command("bench", help="Run benchmark on a model")
def benchmark_model(model: str, input_path: str):
    input_paths = resolve_input_paths(Path(input_path))[:1500]
    if len(input_paths) == 0:
        typer.echo(f"No input files found in '{input_path}'")
        raise typer.Exit(code=1)
    typer.echo(f"Collected input paths: {input_paths}")

    preprocess = PreProcessor()
    class_names = ["" for _ in range(80)] # temporary class names for testing
    postprocess = ObjDetPostprocess(
        model_name="yolov8s",
        model_cfg={"conf_thres": 0.25, "iou_thres": 0.7, "anchors": [None]},
        class_names=class_names,
    )

    asyncio.run(run_inference(model, input_paths, preprocess, postprocess))


if __name__ == "__main__":
    app()