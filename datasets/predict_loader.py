import os
from torch.utils.data import DataLoader
from datasets.GazeFollow import GazeFollow
def test_loader(config,image_list):
    input_size=config.input_size
    output_size=config.output_size
    batch_size=config.batch_size
    num_workers=config.num_workers
    root_dir=config.dataset_dir
    labels = os.path.join(root_dir, "head_information.txt")
    dataset = GazeFollow(root_dir, labels, input_size=input_size, output_size=output_size, is_test_set=True,image_list=image_list)
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return loader