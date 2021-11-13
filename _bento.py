import argparse
import os
import tempfile
import threading

import bentoml
from bentoml.adapters import JsonInput, JsonOutput

from gansketch.train import training_loop
from gansketch.generate import generate

import manta_playground.io as io
import manta_playground.s3 as s3
import manta_playground.env as env


SAVE_BASE_PATH = "example_apps/draw_sentence/results/{}/{}.png"


def upload_eof(storage, s3_dir, id):
    with tempfile.NamedTemporaryFile() as temp:
        filepath = temp.name + ".txt"
        with open(filepath, "wb"):
            pass
        storage.upload(filepath, f"{s3_dir}/{id}/eof")

def upload_callback(storage, params):
    total_iterations = params["iterations"]
    image_num = 0

    def upload_storage(image_path, iteration):
        if iteration % 10 == 1 or iteration == total_iterations:
            s3_key = "".format(iteration)
            storage.upload(image_path, s3_key)
            nonlocal image_num
            image_num += 1

    return upload_storage


def thread_main(params):
    storage = s3.S3Manager(env.S3_BUCKET, env.S3_CREDENTIAL)
    
    with tempfile.TemporaryDirectory() as ckpt_dir:
        best_ckpt = [d for d in os.listdir(ckpt_dir) if d.endswith("_net_G.pth")][-1]
        params.ckpt = os.path.join(ckpt_dir, best_ckpt)

        training_loop(params)
        generate(params, upload_callback(storage, params))

    upload_eof(storage, "example_apps/draw_sketch/results", params["id"])


class SketchDrawingBento(bentoml.BentoService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @bentoml.api(input=JsonInput(), output=JsonOutput())
    def service(self, params):
        trainer = threading.Thread(target=thread_main, kwargs=dict(params=params))
        trainer.daemon = True
        trainer.start()

        return {"success": True}


if __name__ == "__main__":
    bento_svc = SketchDrawingBento()
    # saved_path = bento_svc.service(
    #    {"id": "test", "device": "cuda:0", "prompts": ["draw test sentence in galaxy"]}
    # )

    bento_svc.save()