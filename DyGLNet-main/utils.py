import torch
import torchvision
from dataset import CarvanaDataset,BUSIDataset,PH2Dataset,ISICDataset,BrainMRIDataset,DriveDataset,BCCSDataset,TNBCDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def get_dataloader(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                   train_transform, val_transform, batch_size, num_workers,
                   pin_memory=True, worker_init_fn=None, generator=None):

    train_set = CarvanaDataset(image_dir=train_img_dir, mask_dir=train_mask_dir, transform=train_transform)
    val_set = CarvanaDataset(image_dir=val_img_dir, mask_dir=val_mask_dir, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,shuffle=True,  
    worker_init_fn=worker_init_fn,
    generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,shuffle=False,
    worker_init_fn=worker_init_fn,
    generator=generator)

    return train_loader, val_loader

def get_dataloader_ISIC(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                   train_transform, val_transform, batch_size, num_workers,
                   pin_memory=True, worker_init_fn=None, generator=None):

    train_set =ISICDataset(image_dir=train_img_dir, mask_dir=train_mask_dir, transform=train_transform)
    val_set = ISICDataset(image_dir=val_img_dir, mask_dir=val_mask_dir, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,shuffle=True,  
    worker_init_fn=worker_init_fn,
    generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,shuffle=False,
    worker_init_fn=worker_init_fn,
    generator=generator)

    return train_loader, val_loader

def get_dataloader_BUSI(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                   train_transform, val_transform, batch_size, num_workers,
                   pin_memory=True, worker_init_fn=None, generator=None):

    train_set = BUSIDataset(image_dir=train_img_dir, mask_dir=train_mask_dir, transform=train_transform)
    val_set = BUSIDataset(image_dir=val_img_dir, mask_dir=val_mask_dir, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,shuffle=True,  
    worker_init_fn=worker_init_fn,
    generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,shuffle=False,
    worker_init_fn=worker_init_fn,
    generator=generator)

    return train_loader, val_loader

def get_dataloader_PH2(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                     train_transform, val_transform, batch_size, num_workers,
                     pin_memory=True, worker_init_fn=None, generator=None):

    train_set = PH2Dataset(image_dir=train_img_dir, mask_dir=train_mask_dir, transform=train_transform)
    val_set = PH2Dataset(image_dir=val_img_dir, mask_dir=val_mask_dir, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, 
                            pin_memory=pin_memory, shuffle=True, worker_init_fn=worker_init_fn,
                            generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers,
                          pin_memory=pin_memory, shuffle=False, worker_init_fn=worker_init_fn,
                          generator=generator)
    
    return train_loader, val_loader

def get_dataloader_BrainMRI(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                         train_transform, val_transform, batch_size, num_workers,
                         pin_memory=True, worker_init_fn=None, generator=None):
    
    train_set = BrainMRIDataset(image_dir=train_img_dir, mask_dir=train_mask_dir, transform=train_transform)
    val_set = BrainMRIDataset(image_dir=val_img_dir, mask_dir=val_mask_dir, transform=val_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        generator=generator
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        generator=generator
    )
    
    return train_loader, val_loader


def get_dataloader_DRIVE(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                   train_transform, val_transform, batch_size, num_workers,
                   pin_memory=True, worker_init_fn=None, generator=None):

    train_set = DriveDataset(image_dir=train_img_dir, mask_dir=train_mask_dir, transform=train_transform, train=True)
    val_set = DriveDataset(image_dir=val_img_dir, mask_dir=val_mask_dir, transform=val_transform, train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory, shuffle=True,  
                              worker_init_fn=worker_init_fn, generator=generator)
    
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory, shuffle=False,
                            worker_init_fn=worker_init_fn, generator=generator)

    return train_loader, val_loader


def get_dataloader_BCCS(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                      train_transform, val_transform, batch_size, num_workers,
                      pin_memory=True, worker_init_fn=None, generator=None):
    
    train_set = BCCSDataset(image_dir=train_img_dir, mask_dir=train_mask_dir, transform=train_transform)
    val_set = BCCSDataset(image_dir=val_img_dir, mask_dir=val_mask_dir, transform=val_transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory, shuffle=True, worker_init_fn=worker_init_fn,
                            generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers,
                          pin_memory=pin_memory, shuffle=False, worker_init_fn=worker_init_fn,
                          generator=generator)
    
    return train_loader, val_loader


def get_dataloader_TNBC(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                   train_transform, val_transform, batch_size, num_workers,
                   pin_memory=True, worker_init_fn=None, generator=None):

    train_set = TNBCDataset(image_dir=train_img_dir, mask_dir=train_mask_dir, transform=train_transform)
    val_set = TNBCDataset(image_dir=val_img_dir, mask_dir=val_mask_dir, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,shuffle=True,  
    worker_init_fn=worker_init_fn,
    generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,shuffle=False,
    worker_init_fn=worker_init_fn,
    generator=generator)

    return train_loader, val_loader