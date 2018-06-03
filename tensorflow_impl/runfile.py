import argparse
import os
import sys

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt_file", type=str, default=None, help="results' dir")
    ap.add_argument("--start_layer", type=int, default=None, help="S3 Path. (haim folder is excluded)")
    ap.add_argument("--model", type=str, default=None, help="summary file Path.")

    args = ap.parse_args()

    time_steps_pool = [5]
    lr_pool = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    decay_pool = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
    max_grad_pool = [2.0, 5.0, 7.]
    rms_step_pool = [False, True]
    rms_decay_pool = [False, True]

    param_settings = [(ts,lr,dc,r_step,r_dc, mx_g)  for ts in time_steps_pool \
                                                    for lr in lr_pool \
                                                    for dc in decay_pool \
                                                    for r_step in rms_step_pool \
                                                    for r_dc in rms_decay_pool \
                                                    for mx_g in max_grad_pool]
    for i,(dynamic_time_steps,dynamic_lr,dynamic_decay,dynamic_rms_step,dynamic_rms_decay, max_grad) in enumerate(param_settings):

        print("setting %d/%d\n\n" % (i+1, len(param_settings)))
        print(dynamic_time_steps, dynamic_lr, dynamic_decay, dynamic_rms_step, dynamic_rms_decay)

        cmd = "python model_estimator.py --model " + args.model + \
              " --start_layer " + str(args.start_layer) + \
              " --ckpt_file " + str(args.ckpt_file)  + \
              " --dynamic_eval True --dynamic_time_steps " + str(dynamic_time_steps) + \
              " --dynamic_lr " + str(dynamic_lr) + \
              " --dynamic_decay " + str(dynamic_decay) + \
              " --dynamic_rms_step " + str(dynamic_rms_step) + \
              " --max_grad_norm " + str(max_grad) + \
              " --dynamic_rms_decay " + str(dynamic_rms_decay)

        os.system(cmd)

    return 0

if __name__ == "__main__":
    sys.exit(main())
