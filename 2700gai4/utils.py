from simpletransformers.config.model_args import Seq2SeqArgs
import os
import torch

def load_model_args(input_dir):
    args = Seq2SeqArgs()
    args.load(input_dir)
    return args
def save_model(model_class,logger, output_dir=None, optimizer=None, scheduler=None, model=None, results=None):
        if not output_dir:
            output_dir = model_class.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving model into {output_dir}")

        if model and not model_class.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_class._save_model_args(output_dir)

            if model_class.args.model_type in ["bart", "marian"]:
                os.makedirs(os.path.join(output_dir), exist_ok=True)
                model_to_save.save_pretrained(output_dir)
                model_class.config.save_pretrained(output_dir)
                if model_class.args.model_type == "bart":
                    model_class.encoder_tokenizer.save_pretrained(output_dir)
            else:
                os.makedirs(os.path.join(output_dir, "encoder"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "decoder"), exist_ok=True)
                model_class.encoder_config.save_pretrained(os.path.join(output_dir, "encoder"))
                model_class.decoder_config.save_pretrained(os.path.join(output_dir, "decoder"))

                model_to_save = (
                    model_class.model.encoder.module if hasattr(model_class.model.encoder, "module") else model_class.model.encoder
                )
                model_to_save.save_pretrained(os.path.join(output_dir, "encoder"))

                model_to_save = (
                    model_class.model.decoder.module if hasattr(model_class.model.decoder, "module") else model_class.model.decoder
                )

                model_to_save.save_pretrained(os.path.join(output_dir, "decoder"))

                model_class.encoder_tokenizer.save_pretrained(os.path.join(output_dir, "encoder"))
                model_class.decoder_tokenizer.save_pretrained(os.path.join(output_dir, "decoder"))

            torch.save(model_class.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and model_class.args.save_optimizer_and_scheduler:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))