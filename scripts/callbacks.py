from transformers.trainer_callback import TrainerCallback

class EarlyStoppingCallback(TrainerCallback):
    # any more training, and this overfits on train.
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def on_step_begin(self, args, state, control, **kwargs):  
        
        if len(state.log_history) > 0:
            # get last log history
            last_loss = None
            
            for k in state.log_history[::-1]:
                if "loss" in k:
                    last_loss = k["loss"]
                    break
                    
            if last_loss < self.threshold:
                control.should_training_stop = True

class ResampleCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"
    def __init__(self, collator, model, resample_rate):
        
        self.collator = collator
        self.model = model
        self.resample_rate = resample_rate

        self.last_step_num = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.reset_and_resample(args, state, control, **kwargs)
        
    def on_step_begin(self, args, state, control, **kwargs):
        self.reset_and_resample(args, state, control, **kwargs)

    def reset_and_resample(self, args, state, control, **kwargs):
        step_num = int(state.global_step)

        if self.last_step_num == step_num:
            return
        
        print("STARTING EPOCH: " + str(step_num))

        if self.resample_rate != None and step_num % self.resample_rate == 0:
            self.collator.resample(step=step_num)

        self.last_step_num = step_num
