import argparse
from multiprocessing.connection import Listener
from octo.model.octo_model import OctoModel
import numpy as np
import jax
import time

def eval_model(args):
    model = OctoModel.load_pretrained(args.model_path, args.step)
    
    print("Creating listener")
    listener = Listener(('localhost', 6000), authkey=b'secret password')
    print("Listening")
    conn = listener.accept()
    print('connection accepted from', listener.last_accepted)

    while True:
        observation = conn.recv()
        image = observation["obs"]['rgb']
        input_images = np.stack([image])[None]
    
        wrist = observation["obs"]['wrist']
        wrist_images = np.stack([wrist])[None]
        
        cur_instruction = observation["obs"]['instruction']
        task = model.create_tasks(texts=[cur_instruction])
        
        timestep_pad_mask = np.array([[True]])
            
        observation_octo = {
            'image_primary': input_images,
            'image_wrist': wrist_images,
            'timestep_pad_mask': timestep_pad_mask
        }

        actions = model.sample_actions(
            observation_octo, 
            task, 
            unnormalization_statistics=model.dataset_statistics["action"], 
            rng=jax.random.PRNGKey(0)
        )
        conn.send(np.array(actions[0]))
        
        np.set_printoptions(suppress=True, precision=4)
        np.set_printoptions(linewidth=np.inf)
        print(cur_instruction, np.array(actions[0][0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=False, default=None)
    parser.add_argument("--step", type=int, required=False, default=None)
    parser.add_argument("--m", type=float, required=False, default=0.00001)
    args = parser.parse_args()

    while True:
        try:
            eval_model(args)
        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, shutting down.")
            break
        except Exception as e:
            print("Server crashed with exception:")
            print("Restarting after failure...")
            time.sleep(5)
            continue