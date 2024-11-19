import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader


transform = transforms.Compose([
    # transforms.Resize((300,200)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # convert 0-255 to 0-1 and from np to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return (img, label ,path)


def get_data(path, get_path=False, shuffle=False):
    if get_path == True:
        dataset = ImageFolderWithPaths(path, transform=transform)
    else: 
        dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    if shuffle == True: 
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
    else: 
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

    return loader



