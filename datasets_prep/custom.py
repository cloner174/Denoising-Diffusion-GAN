from torch.utils.data import Dataset
import SimpleITK as sitk
from PIL import Image


class DatasetCustom(Dataset):
    
    def __init__(self, labels_df, transform, class_ ='train'):
        self.class_ = class_
        self.transform = transform
        
        self.__load__(labels_df[labels_df['Class'] == self.class_])
    
    def __load__(self, labels_df):
        images_i = []
        images_path = []
        for i in range(labels_df[labels_df['Class'] == self.class_].shape[0]):
            temp_path = labels_df['Path'][labels_df['Class'] == self.class_].iloc[i]
            number_slices = labels_df['ShapeZiro'][labels_df['Class'] == self.class_].iloc[i]
            for j in range(number_slices):
                images_i.append(j)
                images_path.append(temp_path)
        
        self.images_i = images_i
        self.images_path = images_path
        self.current_image = None
    
    def __getitem__(self, index):
        image_path = self.images_path[index]
        image_i = self.images_i[index]
        if self.current_image is None or image_i >= self.current_image.shape[0]:
            self.current_image = self.phrase_data(image_path)
        else:
            pass
        image = Image.fromarray(self.current_image[image_i])
        image = image.convert("RGB")
        image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.images_path)
    
    @staticmethod
    def phrase_data(path):
        try:
            image = sitk.ReadImage(path)
            return sitk.GetArrayFromImage(image)
        except Exception as e:
            raise Exception(e)