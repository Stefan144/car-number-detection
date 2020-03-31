import imageio
from torch.utils.data import Dataset


class OcrDataset(Dataset):
    def __init__(self, pics_paths, negative_paths, transforms=None):

        self.data = pics_paths
        self.target = []
        for i, pic_path in enumerate(self.data):
            car_number = pic_path.split('/')[-1].split('_')[0]
            self.target.append(car_number)
        self.transforms = transforms

        # adding non-cars
        self.data += negative_paths
        for _ in range(len(negative_paths)):
            self.target.append('')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = imageio.imread(self.data[idx])
        text = self.target[idx]
        img = self.transforms(img)

        return {"image": img,
                "text": text}
