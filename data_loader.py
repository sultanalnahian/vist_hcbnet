import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from vist import VIST


class VistDataset(data.Dataset):
    def __init__(self, image_dir, sis_path, story_keys, single_descs, vocab, transform=None):
        self.image_dir = image_dir
        self.vist = VIST(sis_path, story_keys)
        self.ids = list(self.vist.stories.keys())
        self.vocab = vocab
        self.transform = transform
        self.single_descs = single_descs


    def __getitem__(self, index):
        vist = self.vist
        vocab = self.vocab
        story_id = self.ids[index]

        targets = []
        source_description = []
        images = []
        photo_sequence = []
        album_ids = []

        story = vist.stories[story_id]
        image_formats = ['.jpg', '.gif', '.png', '.bmp']
        for annotation in story:
            storylet_id = annotation["storylet_id"]
            image = Image.new('RGB', (256, 256))
            image_id = annotation["photo_flickr_id"]
            photo_sequence.append(image_id)
            album_ids.append(annotation["album_id"])
            for image_format in image_formats:
                try:
                    image = Image.open(os.path.join(self.image_dir, str(image_id) + image_format)).convert('RGB')
                except Exception:
                    continue

            if self.transform is not None:
                image = self.transform(image)

            images.append(image)

            text = annotation["text"]
            description = self.single_descs[image_id]
            description = description.strip()
            tokens = []
            try:
                tokens = nltk.tokenize.word_tokenize(text.lower())
            except Exception:
                pass

            desc_tokens = []
            try:
                desc_tokens = nltk.tokenize.word_tokenize(description.lower())
            except Exception:
                pass
            
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)
            targets.append(target)
            
            desc_caption = []
            desc_caption.append(vocab('<start>'))
            desc_caption.extend([vocab(token) for token in desc_tokens])
            desc_caption.append(vocab('<end>'))
            src_desc = torch.Tensor(desc_caption)
            source_description.append(src_desc)

        return torch.stack(images), targets, source_description, photo_sequence, album_ids


    def __len__(self):
        return len(self.ids)

    def GetItem(self, index):
        vist = self.vist
        vocab = self.vocab
        story_id = self.ids[index]

        targets = []
        source_description = []
        images = []
        photo_sequence = []
        album_ids = []

        story = vist.stories[story_id]
        image_formats = ['.jpg', '.gif', '.png', '.bmp']
        for annotation in story:
            storylet_id = annotation["storylet_id"]
            image = Image.new('RGB', (256, 256))
            image_id = annotation["photo_flickr_id"]
            photo_sequence.append(image_id)
            album_ids.append(annotation["album_id"])
            for image_format in image_formats:
                try:
                    image = Image.open(os.path.join(self.image_dir, image_id + image_format)).convert('RGB')
                    break
                except Exception:
                    continue

            if self.transform is not None:
                image = self.transform(image)

            images.append(image)

            text = annotation["text"]
            description = self.single_descs[image_id]
            description = description.strip()
            tokens = []
            try:
                tokens = nltk.tokenize.word_tokenize(text.lower())
            except Exception:
                pass
            
            desc_tokens = []
            try:
                desc_tokens = nltk.tokenize.word_tokenize(description.lower())
            except Exception:
                pass

            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)
            targets.append(target)
            
            desc_caption = []
            desc_caption.append(vocab('<start>'))
            desc_caption.extend([vocab(token) for token in desc_tokens])
            desc_caption.append(vocab('<end>'))
            src_desc = torch.Tensor(desc_caption)
            source_description.append(src_desc)

        return images, targets, source_description, photo_sequence, album_ids

    def GetLength(self):
        return len(self.ids)


def collate_fn(data):

    image_stories, caption_stories, source_description, photo_sequence_set, album_ids_set = zip(*data)

    targets_set = []
    lengths_set = []

    for captions in caption_stories:
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        targets_set.append(targets)
        lengths_set.append(lengths)
    
    desc_source_set = []
    desc_lengths_set = []

    for desc in source_description:
        lengths = [len(cap) for cap in desc]
        source = torch.zeros(len(desc), max(lengths)).long()
        for i, cap in enumerate(desc):
            end = lengths[i]
            source[i, :end] = cap[:end]

        desc_source_set.append(source)
        desc_lengths_set.append(lengths)

    return image_stories, targets_set, lengths_set, desc_source_set, desc_lengths_set, photo_sequence_set, album_ids_set


def get_loader(root, sis_path, story_keys, src_descs, vocab, transform, batch_size, shuffle, num_workers):
    vist = VistDataset(image_dir=root, sis_path=sis_path, story_keys = story_keys, single_descs = src_descs, vocab=vocab, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=vist, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader
