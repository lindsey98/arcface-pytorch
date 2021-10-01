from .base import *


class Logo2K(BaseDataset):
    def __init__(self, root, mode, transform=None):
        self.root = root + '/logo2k'
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.classes = range(0, 1171)
            # print(len(set(self.classes)))
        elif self.mode == 'eval':
            self.classes = range(1171, 2340)

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        index = 0
        for i in torchvision.datasets.ImageFolder(root=self.root).imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # print(y)
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                # self.im_paths.append(os.path.join(root, i[0]))
                self.im_paths.append(i[0])
                index += 1

        # print(len(set(self.ys)))


class Logo2K_super(BaseDataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.classes = range(0, 107)
        elif self.mode == 'eval':
            self.classes = range(107, 1276)

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        index = 0
        dataloader = torchvision.datasets.ImageFolder(root=root)
        dataloader.classes = [str(x) for x in sorted(int(d) for d in os.listdir(dataloader.root) if os.path.isdir(os.path.join(dataloader.root, d)))]
        dataloader.class_to_idx = {dataloader.classes[i]: i for i in range(len(dataloader.classes))}
        dataloader.samples = dataloader.make_dataset(dataloader.root, dataloader.class_to_idx, torchvision.datasets.folder.IMG_EXTENSIONS, None)
        dataloader.imgs = dataloader.samples

        for i in dataloader.imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # print(y)
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if int(y) in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                # self.im_paths.append(os.path.join(root, i[0]))
                self.im_paths.append(i[0])
                index += 1

        # print(len(set(self.ys)))