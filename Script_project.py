import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import torchvision
from torchvision import transforms
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, ImageEnhance
import random
import time
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from skimage import transform
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd
import sklearn.model_selection as model_selection
from tqdm import tqdm
from albumentations.augmentations.transforms import CLAHE


########### DATA AUGMENTATION | DATA TRANSFORMATION #######################
IMG_SIZE = 640
image_options = {'resize': True, 'resize_size': IMG_SIZE}

class Resize(object):
    '''
        Resize an image.
    '''
    def __init__(self, output_size):
        assert isinstance(output_size, (int))
        self.output_size = output_size

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        image = Image.fromarray(image)
        image = transforms.Resize(self.output_size, interpolation=2)(image)
        image = np.array(image)

        new_masks = []
        for mask in masks:
            img = Image.fromarray(mask)
            img = transforms.Resize(self.output_size, interpolation=2)(img)
            new_masks.append(np.array(img))

        new_masks = np.array(new_masks)
        new_sample = {'image': image, 'masks': new_masks}
        return new_sample


class RandomRotate90:
    '''
        Randomly rotates an image
    '''
    def __init__(self, num_rot=(1, 2, 3, 4)):
        self.num_rot = num_rot
        self.axes = (0, 1)

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        n = np.random.choice(self.num_rot)
        image_rotate = np.ascontiguousarray(np.rot90(image, n, self.axes))
        new_masks = []
        for (i, mask) in enumerate(masks):
            new_masks.append(np.rot90(mask, n, self.axes))

        new_sample = {'image': image_rotate, 'masks': new_masks}
        return new_sample

class ApplyCLAHE(object):
    '''
        Applies CLAHE (Contrast Limited Adaptative Histogram Equalization)
        transformation to the dataset image.
    '''
    def __init__(self, green=False):
        self.green = green

    def __call__(self, sample):
        light = CLAHE(p=1)
        image, masks = sample['image'], sample['masks']
        image = np.uint8(image)

        image = light(image=image)['image']
        if self.green:
            image = [image[:,:,1]]

        new_sample = {'image': image, 'masks': masks}
        return new_sample


class ImageEnhencer(object):
    '''
        Enhances the brightness/color and contrast of a dataset image
    '''
    def __init__(self, color_jitter=False, green=False):
        self.color_jitter = color_jitter
        self.green = green

    def __call__(self, sample, color_jitter=False):
        t1 = time.time()
        image, masks = sample['image'], sample['masks']
        image = np.uint8(image)
        image = Image.fromarray(image)

        if self.color_jitter:
            image = transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0, hue=0.05)(image)
            image = np.array(image)

        else:
            # light
            enh_bri = ImageEnhance.Brightness(image)
            brightness = 1.3
            image = enh_bri.enhance(brightness)

            # color
            enh_col = ImageEnhance.Color(image)
            color = 1.0
            image = enh_col.enhance(color)

            # contrast
            enh_con = ImageEnhance.Contrast(image)
            contrast = 1.2
            image = enh_con.enhance(contrast)
            image = np.array(image)


        if self.green:
            image = [image[:,:,1]] # green channel

        masks = masks
        new_sample = {'image': image, 'masks': masks}

        return new_sample


class RandomCrop(object):
    """Crop randomly the image in a sample.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        new_masks = []
        for mask in masks:
            mask_crop = mask[top: top + new_h, left: left + new_w]
            new_masks.append(mask_crop)


        new_masks = np.array(new_masks)

        return {'image': image, 'masks': new_masks}


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
    """

    def __init__(self, green=False):
        self.green = green

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']

        image = np.array(image)
        masks = torch.from_numpy(np.array(masks))

        if not self.green:
            image = np.rollaxis(image, 2, 0)

        image = torch.from_numpy(image)

        return {'image': image,
                'masks': masks}

########### DOWNLOAD DATA #######################

def get_files_info(data_path):
    '''
        Gets information about files and folders in data_path.
    '''

    files = os.listdir(data_path)
    print('\t Files info')
    print('Content of the folder : {}'.format(files))
    print('Lenght of the folder : {}'.format(len(files)))

    tasks = ['MA', 'HE', 'EX', 'SE']
    for task in tasks:
        task_path = os.path.join(data_path, 'TrainingSet_masks/' + task + '/')
        task_files = os.listdir(task_path)
        print('Number of train masks for ' + task + ' task :', len(task_files))

    train_images_path = os.path.join(data_path, 'TrainingSet')
    print('Number of train images : %s' % len(os.listdir(train_images_path)))

    test_images_path = os.path.join(data_path, 'TestingSet')
    print('Number of test images : %s' % len(os.listdir(test_images_path)))
    print('-' * 20)
    print('\n')


def load_sitk(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


class IDRiDDataset(Dataset):
    def __init__(self, mode='train', root_dir='Segmentation/',
                 transform=None, tasks=['MA', 'HE', 'EX', 'SE'], data_augmentation=True, green=False):

        super(IDRiDDataset, self).__init__()
        # After resize image
        IMG_SIZE = 640
        if mode == 'train':
            mask_file, image_file = 'TrainingSet_masks/', 'TrainingSet/'

        elif mode == 'val':
            mask_file, image_file = 'TestingSet_masks/', 'TestingSet/'

        else:
            raise EnvironmentError('You should put a valid mode to generate the dataset')

        self.mode = mode
        self.transform = transform
        self.mask_file = mask_file
        self.image_file = image_file
        self.root_dir = root_dir
        self.tasks = tasks
        self.data_augmentation = data_augmentation
        self.process_image = False

    def __len__(self):
        task = self.tasks[0]
        mask_path = os.path.join(self.root_dir, self.mask_file + task)
        return len(os.listdir(mask_path))

    def __getitem__(self, idx):
        'Generate one batch of data'
        sample = self.load(idx)
        return sample

    def load(self, idx):
        masks = []

        if self.mode == 'val':
            idx += 54

        for task in self.tasks:
            suffix = '.tif'
            mask_name = 'IDRiD_{:02d}_'.format(idx + 1) + task + suffix  # if idx = 0. we look for the image 1
            mask_path = os.path.join(self.root_dir, self.mask_file + task + '/' + mask_name)
            mask = load_sitk(mask_path)
            mask = mask[:, :, 0] / 255
            masks.append(mask)

        masks = np.stack(masks, axis=0)

        image_name = 'IDRiD_{:02d}'.format(idx + 1) + '.jpg'
        image_path = os.path.join(self.root_dir, self.image_file + image_name)
        image = load_sitk(image_path)

        masks = masks.astype(np.int16)



        # Define output sample
        sample = {'image': image, 'masks': masks}

        if self.transform:
            sample = self.transform(sample)

        return sample


def load_train_val_data(tasks=['MA', 'HE', 'EX', 'SE'], data_path='Segmentation/', batch_size=8, green=False):
    get_files_info(data_path)


    n_classes = len(tasks)
    n_channels = 3


    transforms_list = [
                       Resize(520), #resize to 520x782
                       RandomCrop(512),
                       RandomRotate90(),
                       ImageEnhencer(color_jitter=True, green=False),
                       ApplyCLAHE(green=green),
                       ToTensor(green=green)]

    ## Image (512, 512, 3)
    transformation = torchvision.transforms.Compose(transforms_list)

    print('\t Loading Train and Validation Datasets... \n')
    train_data = IDRiDDataset(tasks=tasks, transform=transformation, root_dir=data_path)
    val_data = IDRiDDataset(mode='val', tasks=tasks, transform=transformation, root_dir=data_path)

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)

    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True)

    print('Length of train dataset: ', len(train_loader.dataset))
    print('-' * 20)
    print('\n')

    return train_loader, val_loader



#######
train_loader, val_loader = load_train_val_data(tasks=['MA','EX'], data_path='C:/Users/Marta/Desktop/Segmentation.nosync/', green=False)
#train_loader, val_loader = load_train_val_data(tasks=['MA','EX'], data_path='C:/Users/Marta/Desktop/Segmentation.nosync/', green=True)


########### UNET #######################
import torch.nn as nn
import torch

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)
        return self.nConvs(x)



class UNet(nn.Module):
  def __init__(self, n_channels, n_classes):
    '''
    n_channels : number of channels of the input.
                    By default 3, because we have RGB images
    n_labels : number of channels of the ouput.
                  By default 3 (2 labels + 1 for the background)
    '''
    super(UNet, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.inc = ConvBatchNorm(n_channels, 64)
    self.down1 = DownBlock(64, 128, nb_Conv=2)
    self.down2 = DownBlock(128, 256, nb_Conv=2)
    self.down3 = DownBlock(256, 512, nb_Conv=2)
    self.down4 = DownBlock(512, 512, nb_Conv=2)
    self.up1 = UpBlock(1024, 256, nb_Conv=2)
    self.up2 = UpBlock(512, 128, nb_Conv=2)
    self.up3 = UpBlock(256, 64, nb_Conv=2)
    self.up4 = UpBlock(128, 64, nb_Conv=2)
    self.outc = nn.Conv2d(64, n_classes, kernel_size=3, stride=1, padding=1)
    self.last_activation = get_activation('Softmax')


  def forward(self, x):
    x = x.float()
    x1 = self.inc(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)
    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    logits = self.last_activation(self.outc(x))

    return logits


def tensor_to_image(x):
    '''Returns an array of shape CxHxW from a given tensor with shape HxWxC'''

    x = np.rollaxis(x.int().detach().cpu().numpy(), 0, 3)
    return x


def plot(image, masks=None, pred_masks=None):
    '''plots for a given image the ground truth mask and the corresponding predicted mask
      masks: tensor of shape (n_tasks, 512, 512)
    '''
    fig, ax = plt.subplots(1, 3, gridspec_kw={'wspace': 0.15, 'hspace': 0.2,
                                              'top': 0.85, 'bottom': 0.1,
                                              'left': 0.05, 'right': 0.95})

    ax[0].imshow(image.int().detach().cpu().numpy()[0])
    ax[0].axis('off')

    if masks is not None:
        ax[1].imshow(masks[0], cmap='gray')
        ax[1].axis('off')

    if pred_masks is not None:

        thresh = 0.1
        prediction = pred_masks[0].detach().cpu().numpy()
        max_prob = np.max(prediction)
        img_pred = np.zeros(prediction.shape)
        img_pred[prediction >= thresh * max_prob] = 1
        ax[2].imshow(img_pred, cmap='gray')
        ax[2].axis('off')

    ax[0].set_title('Original Image')
    ax[1].set_title('Ground Truth Seg EX')
    ax[2].set_title('Predicted Seg EX')

    fig.canvas.draw()

    return fig

##########METRICS##############
def AUPR(mask, prediction):
    '''Computes the Area under Precision-Recall Curve for a given ground-truth mask and predicted mask'''
    threshold_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    precisions = []
    recalls = []

    for thresh in threshold_list:
        thresh_pred = np.zeros(prediction.shape)
        thresh_pred[prediction >= thresh] = 1


        P = np.count_nonzero(mask)
        TP = np.count_nonzero(mask * thresh_pred)
        FP = np.count_nonzero(thresh_pred - (mask * thresh_pred))

        if (P > 0) and (TP + FP > 0):
            precision = TP * 1.0 / (TP + FP)
            recall = TP * 1.0 / P
        else:
            precision = 1
            recall = 0

        precisions.append(precision)
        recalls.append(recall)

    return auc(recalls, precisions)

def aupr_on_batch(masks, pred):
    '''Computes the mean AUPR over a batch during training'''
    auprs = []
    for i in range(pred.shape[0]):
        prediction = pred[i][0].cpu().detach().numpy()
        mask = masks[i].cpu().detach().numpy()
        auprs.append(AUPR(mask, prediction))

    return np.mean(auprs)

def auc_on_batch(masks, pred):
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    aucs = []
    for i in range(pred.shape[0]):
        prediction = pred[i][0].numpy()
        mask = masks[i].numpy()
        aucs.append(roc_auc_score(mask.reshape(-1), prediction.reshape(-1)))
    return np.mean(aucs)


########### SAVE MODELS #######################
import tensorflow as tf

main_path = 'C:/Users/Marta/Desktop/Segmentation.nosync/'
sets_path = os.path.join(main_path, 'datasets/')
csv_path = os.path.join(main_path, 'data/tumor_count.csv')
data_folder = os.path.join(main_path, 'data/')


save_path_ex_rgb_1 = 'C:/Users/Marta/Desktop/content/save_ex_rgb_1/'
save_path_ex_rgb_2 = 'C:/Users/Marta/Desktop/content/save_ex_rgb_2/'
save_path_ex_g_1 = 'C:/Users/Marta/Desktop/content/save_ex_g_1/'
save_path_ex_g_2 = 'C:/Users/Marta/Desktop/content/save_ex_g_2/'
save_path_ma_rgb_1 = 'C:/Users/Marta/Desktop/content/save_ma_rgb_1/'
save_path_ma_rgb_2 = 'C:/Users/Marta/Desktop/content/save_ma_rgb_2/'
save_path_ma_g_1 = 'C:/Users/Marta/Desktop/content/save_ma_g_1/'
save_path_ma_g_2 = 'C:/Users/Marta/Desktop/content/save_ma_g_2/'

session_name = 'Test_session' + '_' + time.strftime('%m.%d %Hh%M')

model_path_ex_rgb_1 = 'C:/Users/Marta/Desktop/content/save_ex_rgb_1/' + 'models/' + session_name + '/'
model_path_ex_rgb_2 = 'C:/Users/Marta/Desktop/content/save_ex_rgb_2/' + 'models/' + session_name + '/'
model_path_ex_g_1 = 'C:/Users/Marta/Desktop/content/save_ex_g_1/' + 'models/' + session_name + '/'
model_path_ex_g_2 = 'C:/Users/Marta/Desktop/content/save_ex_g_2/' + 'models/' + session_name + '/'
model_path_ma_rgb_1 = 'C:/Users/Marta/Desktop/content/save_ma_rgb_1/' + 'models/' + session_name + '/'
model_path_ma_rgb_2 = 'C:/Users/Marta/Desktop/content/save_ma_rgb_2/' + 'models/' + session_name + '/'
model_path_ma_g_1 = 'C:/Users/Marta/Desktop/content/save_ma_g_1/' + 'models/' + session_name + '/'
model_path_ma_g_2 = 'C:/Users/Marta/Desktop/content/save_ma_g_2/' + 'models/' + session_name + '/'

tensorboard_folder_ex_rgb_1 = 'C:/Users/Marta/Desktop/content/save_ex_rgb_1/' + 'tensorboard_logs/'
tensorboard_folder_ex_rgb_2 = 'C:/Users/Marta/Desktop/content/save_ex_rgb_2/'+ 'tensorboard_logs/'
tensorboard_folder_ex_g_1 = 'C:/Users/Marta/Desktop/content/save_ex_g_1/'+ 'tensorboard_logs/'
tensorboard_folder_ex_g_2 = 'C:/Users/Marta/Desktop/content/save_ex_g_2/'+ 'tensorboard_logs/'
tensorboard_folder_ma_rgb_1 = 'C:/Users/Marta/Desktop/content/save_ma_rgb_1/'+ 'tensorboard_logs/'
tensorboard_folder_ma_rgb_2 = 'C:/Users/Marta/Desktop/content/save_ma_rgb_2/'+ 'tensorboard_logs/'
tensorboard_folder_ma_g_1 = 'C:/Users/Marta/Desktop/content/save_ma_g_1/'+ 'tensorboard_logs/'
tensorboard_folder_ma_g_2 = 'C:/Users/Marta/Desktop/content/save_ma_g_2/'+ 'tensorboard_logs/'


loss_function = 'dice_loss'



## PARAMETERS OF THE MODEL
learning_rate = 1e-4
image_size = (512, 512)
n_labels = 2
epochs = 10
batch_size = 8
print_frequency = 1
save_frequency = 10
save_model = True
tumor_percentage = 0.5
tensorboard = True

device = torch.device('cpu')


def to_var(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.to(device)
    return x

def to_numpy(x):
    if not (isinstance(x, np.ndarray) or x is None):
        if x.is_cuda:
            x = x.data.cpu()
        x = x.numpy()
    return x

def print_summary(epoch, i, nb_batch, loss, loss_name ,batch_time,
                  average_loss, average_time, auc, aupr, mode):
    '''
        mode = Train or Test
    '''
    summary = '[' + str(mode) + '] Epoch: [{0}][{1}/{2}]\t'.format(
        epoch, i, nb_batch)

    string = ''
    string += '{} : {:.4f} '.format(loss_name, loss)
    string += '(Average {:.4f}) '.format(average_loss)
    string += 'AUC {:.4f} '.format(auc)
    string += 'AUPR {:.4f} \t'.format(aupr)
    string += 'Batch Time {:.4f} '.format(batch_time)
    string += '(Average {:.4f}) \t'.format(average_time)



    summary += string

    print(summary)


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']
    best_model = state['best_model']
    val_loss = state['val_loss']
    model = state['model']
    loss = state['loss']

    if best_model:
        filename = save_path + '/' + \
                    'best_model.{}--{}.pth.tar'.format(loss, model)
    else:
        filename = save_path + '/' + \
                  'model.{}--{}--{:02d}.pth.tar'.format(loss, model, epoch)

    torch.save(state, filename)

def focal_loss(input: torch.Tensor, target: torch.Tensor, gamma=2, alpha=1):
    """Focal loss (auto-weighted cross entropy variant).

    See: https://arxiv.org/pdf/1708.02002.pdf
    """
    target = target.detach()
    ce_loss = F.cross_entropy(input, target, reduce=False)  # vector of loss terms
    loss = alpha * (1 - torch.exp(-ce_loss)) ** gamma * ce_loss
    return loss.mean()

def dice_loss(input, target):
    smooth = 1.
    target = target.float()
    input = input.float()
    input_flat = input.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (input_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) /
                (input_flat.pow(2).sum() + target_flat.pow(2).sum() + smooth))


def mean_dice_loss(input, target):
    channels = list(range(target.shape[1]))
    loss = 0
    for channel in channels:
        dice = dice_loss(input[:, channel, ...],
                         target[:, channel, ...])
        loss += dice

    return loss / len(channels)


def BCE_(y_pred, y_true):
    class_weights = tf.constant([8],dtype=tf.float32)
    tensor_one = tf.constant([1],dtype=tf.float32)

    pred_flat = tf.reshape(y_pred.cpu(), [-1, 1])
    true_flat = tf.reshape(y_true.cpu(), [-1, 1])

    weight_map = tf.multiply(true_flat, class_weights)
    weight_map = tf.add(weight_map, tensor_one)

    loss_map = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_flat, labels=true_flat)
    loss_map = tf.multiply(loss_map, weight_map)
    loss = tf.reduce_mean(loss_map)
    return loss

def weighted_BCELoss(output, target, weights=[5, 1]):

    output = output.clamp(min=1e-5, max=1-1e-5)
    if weights is not None:
        assert len(weights) == 2

        loss = -weights[0] * (target * torch.log(output)) - weights[1] * ((1 - target) * torch.log(1 - output))
    else:
        loss = -target * torch.log(output) - (1 - target) * torch.log(1 - output)

    return torch.mean(loss)



def train_loop(loader, model, criterion, optimizer, writer, epoch, lr_scheduler=None, model_type='UNet'):
    logging_mode = 'Train' if model.training else 'Val'

    end = time.time()
    time_sum, loss_sum = 0, 0
    train_aupr, train_auc = 0.0, 0.0
    aupr_sum, auc_sum = 0.0, 0.0

    auprs = []
    for (i, sample) in enumerate(loader, 1):
        images, masks = sample['image'], sample['masks']
        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__


        model.to(device)
        images, masks = to_var(images.float(), device), to_var(masks.float(), device)

        # compute output
        output = model(images)

        # Compute loss
        if model_type == 'UNet':
            if loss_name == 'BCELoss':
                loss = criterion(output.view(images.shape[0], -1), masks.view(images.shape[0], -1))
            else:
                loss = criterion(output, masks) # Loss


        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute AUC scores

        if loss_name == 'BCELoss':
            thresh = Variable(torch.Tensor([0.1]))
            pred_masks = (output > thresh).float()
        else:
            pred_masks = output
        train_auc = auc_on_batch(masks, pred_masks)
        train_aupr = aupr_on_batch(masks, pred_masks)
        auprs.append(train_aupr)


        batch_time = time.time() - end

        time_sum += batch_size * batch_time
        loss_sum += batch_size * loss
        average_loss = loss_sum / (i * batch_size)
        average_time = time_sum / (i * batch_size)

        end = time.time()

        if i % print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), loss, loss_name, batch_time,
                          average_loss, average_time, train_auc, train_aupr, logging_mode)
        if tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, loss.item(), step)

            writer.add_scalar(logging_mode + '_auc', train_auc, step)
            writer.add_scalar(logging_mode + '_aupr', train_aupr, step)


    if tensorboard:
        n = images.shape[0]
        masks = to_numpy(masks)

        for batch in range(n):
            fig = plot(images[batch],
                       masks[batch], pred_masks[batch])

            writer.add_figure(logging_mode + str(batch),
                              fig, epoch)

    if lr_scheduler is not None:
        lr_scheduler.step(loss)

    mean_aupr = np.mean(auprs)
    return loss, mean_aupr


def main_loop(tasks, criterion, model_path, save_path, tensorboard_folder, data_path=main_path, batch_size=batch_size, model_type='UNet', green=False):


    data_path = data_path
    n_labels = len(tasks)
    n_channels = 1 if green else 3 # green or RGB
    train_loader, val_loader = load_train_val_data(tasks=tasks, data_path=data_path, batch_size=batch_size, green=green)

    if model_type == 'UNet':
        lr = learning_rate
        model = UNet(n_channels, n_labels)
        # Choose loss function
        #criterion = nn.MSELoss()
        # criterion = dice_loss


    else:
        raise TypeError('Please enter a valid name for the model type')

    try:
        loss_name = criterion._get_name()
    except AttributeError:
        loss_name = criterion.__name__

    if loss_name == 'BCEWithLogitsLoss':
        lr = 1e-4
        print('learning rate: ', lr)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=7)

    if tensorboard:
        log_dir = tensorboard_folder + session_name + '/'
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    else:
        writer = None

    max_aupr = 0.0
    for epoch in range(epochs):
        print('******** Epoch [{}/{}]  ********'.format(epoch + 1, epochs + 1))
        print(session_name)

        # train for one epoch
        model.train(True)
        print('Training with batch size : ', batch_size)
        train_loop(train_loader, model, criterion, optimizer, writer, epoch,
                   lr_scheduler=lr_scheduler,
                   model_type=model_type)

        # evaluate on validation set
        print('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_aupr = train_loop(val_loader, model, criterion,
                                  optimizer, writer, epoch)

        # Save best model
        if val_aupr > max_aupr and epoch > 3:
            print('\t Saving best model, mean aupr on validation set: {:.4f}'.format(val_aupr))
            max_aupr = val_aupr
            save_checkpoint({'epoch': epoch,
                             'best_model': True,
                             'model': model_type,
                             'state_dict': model.state_dict(),
                             'val_loss': val_loss,
                             'loss': loss_name,
                             'optimizer': optimizer.state_dict()}, model_path)

        elif save_model and (epoch + 1) % save_frequency == 0:
            save_checkpoint({'epoch': epoch,
                             'best_model': False,
                             'model': model_type,
                             'loss': loss_name,
                             'state_dict': model.state_dict(),
                             'val_loss': val_loss,
                             'optimizer': optimizer.state_dict()}, model_path)


    return model

######RUN MODELS FOR EX - GREEN VS RGB - MSE VS DICE_LOSS######
model_EX_1_g = main_loop(tasks=["EX"],criterion = nn.MSELoss(),save_path= save_path_ex_g_1, model_path = model_path_ex_g_1, tensorboard_folder = tensorboard_folder_ex_g_1, data_path=main_path, green=True)

model_EX_2_g = main_loop(tasks=["EX"],criterion = dice_loss,save_path= save_path_ex_g_2, model_path = model_path_ex_g_2, tensorboard_folder = tensorboard_folder_ex_g_2, data_path=main_path, green=True)

model_EX_1_rgb = main_loop(tasks=["EX"],criterion = nn.MSELoss(),save_path= save_path_ex_rgb_1, model_path = model_path_ex_rgb_1, tensorboard_folder = tensorboard_folder_ex_rgb_1, data_path=main_path, green=False)

model_EX_2_rgb = main_loop(tasks=["EX"],criterion = dice_loss,save_path= save_path_ex_rgb_2, model_path = model_path_ex_rgb_2, tensorboard_folder = tensorboard_folder_ex_rgb_2, data_path=main_path, green=False)



model_MA_1_g = main_loop(tasks=["MA"],criterion = nn.MSELoss(),save_path= save_path_ma_g_1, model_path = model_path_ma_g_1, tensorboard_folder = tensorboard_folder_ma_g_1, data_path=main_path, green=True)

model_MA_2_g = main_loop(tasks=["MA"],criterion = dice_loss,save_path= save_path_ma_g_2, model_path = model_path_ma_g_2, tensorboard_folder = tensorboard_folder_ma_g_2, data_path=main_path, green=True)

model_MA_1_rgb = main_loop(tasks=["MA"],criterion = nn.MSELoss(),save_path= save_path_ma_rgb_1, model_path = model_path_ma_rgb_1, tensorboard_folder = tensorboard_folder_ma_rgb_1, data_path=main_path, green=False)

model_MA_2_rgb = main_loop(tasks=["MA"],criterion = dice_loss,save_path= save_path_ma_rgb_2, model_path = model_path_ma_rgb_2, tensorboard_folder = tensorboard_folder_ma_rgb_2, data_path=main_path, green=False)



sample = val_loader.__iter__().next()
image, masks = sample['image'], sample['masks']



############ROC-AUC FOR EACH MODEL#############
checkpoint = torch.load(save_path_ex_g_1 + 'models/' + 'Test_session_01.18 16h01/'+ 'model.MSELoss--UNet--09.pth.tar')
model = UNet(1, 1)
model.load_state_dict(checkpoint['state_dict'])
pred_mask = model(image)
batch = 0
mask = masks[batch][0].cpu().detach().numpy()
prediction = pred_mask[batch][0].cpu().detach().numpy()
print('ROC AUC: ', roc_auc_score(mask.reshape(-1), prediction.reshape(-1)))


checkpoint_1 = torch.load(save_path_ex_g_2 + 'models/' + 'Test_session_01.18 18h01/'+ 'model.dice_loss--UNet--09.pth.tar')
model_1 = UNet(1, 1)
model_1.load_state_dict(checkpoint_1['state_dict'])
pred_mask_1 = model_1(image)
batch = 0
mask = masks[batch][0].cpu().detach().numpy()
prediction_1 = pred_mask_1[batch][0].cpu().detach().numpy()
print('ROC AUC: ', roc_auc_score(mask.reshape(-1), prediction_1.reshape(-1)))


checkpoint_2 = torch.load(save_path_ex_rgb_1 + 'models/' + 'Test_session_01.18 18h01/'+ 'model.MSELoss--UNet--09.pth.tar')
model_2 = UNet(3, 1)
model_2.load_state_dict(checkpoint_2['state_dict'])
pred_mask_2 = model_2(image)
batch = 0
mask = masks[batch][0].cpu().detach().numpy()
prediction_2 = pred_mask_2[batch][0].cpu().detach().numpy()
print('ROC AUC: ', roc_auc_score(mask.reshape(-1), prediction_2.reshape(-1)))


checkpoint_3 = torch.load(save_path_ex_rgb_2 + 'models/' + 'Test_session_01.18 18h01/'+ 'model.dice_loss--UNet--09.pth.tar')
model_3 = UNet(3, 1)
model_3.load_state_dict(checkpoint_3['state_dict'])
pred_mask_3 = model_3(image)
batch = 0
mask = masks[batch][0].cpu().detach().numpy()
prediction_3 = pred_mask_3[batch][0].cpu().detach().numpy()
print('ROC AUC: ', roc_auc_score(mask.reshape(-1), prediction_3.reshape(-1)))


############PLOT RESULTS#############
#import matplotlib.pyplot as plt
pred_mask = model_1(image)
x=plot(image[7],masks[7], pred_mask[7])
x.savefig('7.png')

pred_mask = model_1(image)
x=plot(image[3],masks[3], pred_mask[3])
x.savefig('3.png')

pred_mask = model_1(image)
x=plot(image[5],masks[5], pred_mask[5])
x.savefig('5.png')