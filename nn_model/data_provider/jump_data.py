import numpy as np
import os
import cv2

class JumpData:
    def __init__(self):
        self.data_dir = '/home/anjie/projects/we_jump/data'
        self.name_list = []
        self.get_name_list()
        self.val_name_list = self.name_list[:50]
        self.train_name_list = self.name_list[50:]

    def get_name_list(self):
        for i in range(3, 7):
            dir = os.path.join(self.data_dir, 'exp_%02d' % i)
            this_name = os.listdir(dir)
            this_name = [os.path.join(dir, name) for name in this_name]
            self.name_list = self.name_list + this_name
        self.name_list_raw = self.name_list
        self.name_list = filter(lambda name: 'res' in name, self.name_list)
        self.name_list = list(self.name_list)

        def _name_checker(name):
            posi = name.index('_res')
            img_name = name[:posi] + '.png'
            if img_name in self.name_list_raw:
                return True
            else:
                return False

        self.name_list = list(filter(_name_checker, self.name_list))


    def next_batch(self, batch_size=8):
        batch_name = np.random.choice(self.train_name_list, batch_size)
        batch = {}
        for idx, name in enumerate(batch_name):
            posi = name.index('_res')
            img_name = name[:posi] + '.png'
            x, y = name[name.index('_h_') + 3: name.index('_h_') + 6], name[name.index('_w_') + 3: name.index('_w_') + 6]
            x, y = int(x), int(y)
            img = cv2.imread(img_name)
            label = np.array([x, y], dtype=np.float32)
            if idx == 0:
                batch['img'] = img[np.newaxis, :, :, :]
                batch['label'] = label.reshape([1, label.shape[0]])
            else:
                img_tmp = img[np.newaxis, :, :, :]
                label_tmp = label.reshape((1, label.shape[0]))
                batch['img'] = np.concatenate((batch['img'], img_tmp), axis=0)
                batch['label'] = np.concatenate((batch['label'], label_tmp), axis=0)
        return batch


if __name__ == '__main__':
    data = JumpData()
    data.next_batch(4)
