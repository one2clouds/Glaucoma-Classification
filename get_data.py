import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader



class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return (img, label ,path)


def get_data(path, data_used, get_path=False, shuffle=False):
    if data_used == 'eyepacs_airogs':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # convert 0-255 to 0-1 and from np to tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    elif data_used == 'dristi_gs1':
        transform = transforms.Compose([
            transforms.RandomCrop(size=(1850,1850), pad_if_needed=True), 
            transforms.Resize(size=(512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # convert 0-255 to 0-1 and from np to tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        print("data used is not given")
        

    if get_path == True:
        dataset = ImageFolderWithPaths(path, transform=transform)
    else: 
        dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    # for data in dataset:
    #     print(data[0].shape)

    # print(dataset[0]) # Because of this I got to know the Glaucoma was labelled as 0 and normal as 1, that's why the labels were flipped before
    # exit()

    # for data in dataset:
    #     print(data[0].shape)

    if shuffle == True: 
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
    else: 
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

    return loader



