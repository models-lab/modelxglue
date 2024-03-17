# Docker in docker: https://devopscube.com/run-docker-in-docker/
# Codabench: https://www.codabench.org/
import itertools
import os
import re

import docker

from docker import APIClient
from docker.errors import BuildError
from docker.utils.json_stream import json_stream


def run_docker(root: str, shared_folder={}, command=None):
    image_id = build_image(root)
    client = docker.from_env()
    volume_map = {}

    if command is None:
        raise Exception("No command specified")

    if shared_folder is not None:
        volume_map = [dict({os.path.abspath(src): {'bind': tgt, 'mode': 'rw'}}) for src, tgt in shared_folder.items()][0]
        #volume_map = {os.path.abspath(shared_folder): {'bind': '/shared', 'mode': 'rw'}}

    current_user = os.getuid()

    # the actual thing to call has to be encoded in the configuration file
    container = client.containers.run(image_id, command,
                                      #user=current_user,
                                      #group_add=[os.getgid()],
                                      volumes=volume_map, detach=True)

    process = container.logs(stream=True, follow=True)
    print('Stream logging the container..')
    for line in process:
        print(line)

    # Check return code of the container process
    exit_code = container.wait()['StatusCode']
    print('Container exited with code', exit_code)
    if exit_code != 0:
        raise Exception("Container exited with code", exit_code)

def build_image(root: str):
    # Uses the low-level API to build an image from a Dockerfile. This is needed to make sure that the output
    # is shown as the image is being built.
    # The code is based on the images.py#build() method in the docker-py library.

    # dockerfile = root + '/Dockerfile'
    print("Running docker... ", root)
    cli = APIClient()

    response = cli.build(
        #nocache=True,
        path=root,
        #tag="repro:1"  # rm=True, tag='yourname/volume'
    )

    image_id = None
    result_stream, internal_stream = itertools.tee(json_stream(response))
    for chunk in result_stream:
        if 'error' in chunk:
            raise BuildError(chunk['error'], result_stream)
        if 'stream' in chunk:
            match = re.search(
                r'(^Successfully built |sha256:)([0-9a-f]+)$',
                chunk['stream']
            )
            if match:
                image_id = match.group(2)

            print(chunk['stream'].strip())
        # last_event = chunk

    print("Build image with id: ", image_id)

    #    generator = response
    #    for item in generator:
    #        print(item)

    # print(logs)
    return image_id
