
# =============================================================================
# Save bad examples (not random yet) TODO
# =============================================================================
def save_wrong_images(self, configs, batch, ground_truth, model_output):
    # Saves ALL wrong classified images of RANDOM validation steps
    if not os.path.exists("results/wrong_examples/"):
        os.makedirs("results/wrong_examples/")
    for index, image in enumerate(batch['image']):
        for key in keys:
            if batch['reader_phase_number'] != model_output:
                im = transforms.ToPILImage()(batch['images'][index].cpu()).convert("RGB")
                im.save(f"results/wrong_examples/{configs.name}_step_idx{index}.jpg")