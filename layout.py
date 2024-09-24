import os
from os.path import basename, join

import pytesseract
from PIL import Image


def crop_words(image_path, layout_file, word_folder):
    with open(layout_file, 'r') as f:
        a = f.read().strip().split('\n')
    a = [list(map(int, i.strip(' ,').split(','))) for i in a]
    a = [i for i in a if len(i) == 5]
    img = Image.open(image_path).convert('RGB')
    for idx,i in enumerate(a):
        img.crop((
            i[0], i[1],
            i[0]+i[2], i[1]+i[3]
        )).save(join(word_folder, f'{idx}.jpg'))

def main(image_path, pretrained, output):
    try:
        if os.path.exists(pretrained) and os.path.isdir(pretrained):
            os.environ['TESSDATA_PREFIX'] = pretrained
        else:
            raise ValueError(f'Please check the openseg pretrained folder path: {pretrained}')
        print(f'Processing OpenSeg for file: {basename(image_path)}')
        results = pytesseract.image_to_data(
            image_path,
            output_type=pytesseract.Output.DICT,
            # Using multiple layout traineddata files for processing.
            # If a particular layout file is not present, it will be ignored
            lang='layout1+layout2+layout3+layout4'
        )

        regions = []
        for i in range(0, len(results['text'])):
            if int(results['conf'][i]) <= 0:
                # Skipping the region as confidence is too low
                continue
            x = results['left'][i]
            y = results['top'][i]
            w = results['width'][i]
            h = results['height'][i]
            if h < 10:
                # Skipping box as height is too low.
                continue
            regions.append(','.join(list(map(str, [
                x, y, w, h,
                results['line_num'][i] + 1,
            ]))))
        outfile = join(output, 'layout.txt')
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write('\n'.join(regions))
        word_folder = join(output, 'words')
        if not os.path.exists(word_folder):
            os.makedirs(word_folder)
        crop_words(image_path, outfile, word_folder)
        return word_folder
    except Exception as e:
        print(f"Error processing file {image_path}: {e}")