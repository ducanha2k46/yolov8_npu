import time
import cv2
import typer
import asyncio
import multiprocessing as mp

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

async def submit_with(submitter, input_queues, num_target = 5000):
    idx = 0
    run = [True for _ in range(5)]
    end = 0
    while end != 5:
        for id_ in range(5):
            if not run[id_]:
                continue
            model_input, context = input_queues[id_].get()
            if model_input is None:
                end += 1
                run[id_] = False
                continue
            await submitter.submit(model_input, context=(context, idx%5, ))
        idx += 1

async def recv_with(receiver, output_queues):
    while True:
        try:
            async def recv():
                context, outputs = await receiver.recv()
                return context, outputs

            task = asyncio.create_task(recv())
            (context, idx), outputs = await asyncio.wait_for(task, timeout=0.1)
            output_queues[idx].put((outputs, context))
        except asyncio.TimeoutError:  
            break

    return None

async def warboy_inference(submitter, receiver, input_queues, output_queues):
    model_ouputs = []
    submit_task = asyncio.create_task(submit_with(submitter, input_queues))
    recv_task = asyncio.create_task(recv_with(receiver, output_queues))
    await submit_task
    await recv_task
    return None

def preprocess_task(input_paths: str, preprocess:Callable, input_queue: mp.Queue, img_queue: mp.Queue, idx: int):
    id_ = 0
    for input_path in tqdm(input_paths, desc="Preprocess"):
        if id_ % 5 == idx:
            img = cv2.imread(input_path)
            img_queue.put(img)
            input_queue.put(preprocess(img))
        id_ += 1
    
    img_queue.put(None)
    input_queue.put((None, None))
    return

def postprocess_task(postprocess: Callable, output_queue: mp.Queue, img_queue: mp.Queue, num_target=5000):
    idx = 0
    last_idx = 0
    while True:
        input_img = img_queue.get()
        if input_img is None:
            break
        output, context = output_queue.get()
        postprocess(output, context, input_img)
        idx += 1

async def run_inference(
    model: str,
    input_paths: str,
    preprocess: Callable,
    postprocess: Callable,
    device_str:str = "warboy(2)*1"
):
    warning = """WARN: the benchmark results may depend on the number of input samples,sizes of the images, and a machine where this benchmark is running."""
    queries = len(input_paths)
    print(f"Run benchmark on {queries} input samples ...")
    print(decorate_with_bar(warning))
    input_queues = [mp.Queue(maxsize=100) for _ in range(5)]
    img_queues = [mp.Queue(maxsize=100) for _ in range(5)]
    output_queues = [mp.Queue(maxsize=100) for _ in range(5)]

    preprocs = [mp.Process(target = preprocess_task, args=(input_paths, preprocess, input_queues[idx], img_queues[idx], idx)) for idx in range(5)]
    postprocs = [mp.Process(target=postprocess_task, args=(postprocess, output_queues[idx], img_queues[idx])) for idx in range(5)]

    initial_time = time.perf_counter()
    async with create_queue(model = model, device=device_str, worker_num = 32) as (submitter, receiver):
        for preproc in preprocs:
            preproc.start()
        for postproc in postprocs:
            postproc.start()
        outputs = await warboy_inference(submitter, receiver, input_queues, output_queues)
        for preproc in preprocs:
            preproc.join()
        for postproc in postprocs:
            postproc.join()
    all_done = time.perf_counter()
    print(
        decorate_result(
            all_done - initial_time, queries, "Preprocess -> Inference -> Postprocess"
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
    input_paths = resolve_input_paths(Path(input_path))
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