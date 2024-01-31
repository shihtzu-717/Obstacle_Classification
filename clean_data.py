from pathlib import Path

ANNOTATIONS = '/annotations/'
IMAGES = '/images/'
LABEL_LIST = ['positive', 'negative']

def find_image(annotation: Path) -> Path:
    ext_list = ['.jpg', '.png']
    annotation = str(annotation).replace(ANNOTATIONS, IMAGES, 1)
    annotation = Path(annotation)
    for ext in ext_list:
        image = annotation.with_suffix(ext)
        if image.exists():
            return image
    raise Exception(f'image not found: {annotation}')

def find_label(annotation_path: Path) -> str:
    for part in annotation_path.parts:
        if part in LABEL_LIST:
            return part
    raise Exception(f'label not found: {annotation_path}')

def make_out_path(annotation_path: Path, data_root: Path, out_path: Path) -> Path:
    annotation_path = annotation_path.relative_to(data_root)
    label = find_label(annotation_path)
    annotation_path = str(annotation_path)
    annotation_path = annotation_path.replace(f'/{label}/', '/', 1)
    annotation_path = annotation_path.replace(ANNOTATIONS, '/', 1)

    out_annotation_path = out_path / label / annotation_path
    if not out_annotation_path.parent.exists():
        out_annotation_path.parent.mkdir(parents=True, exist_ok=True)
    return out_annotation_path

def main() -> None:
    data_root = Path('/home/myunchul/data/dareesoft/pothole_data/')
    out_path = Path('/nas/dareesoft/pothole_data')

    class_set = set()
    txt_cnt = 0
    line_cnt = 0
    for annotation_path in data_root.glob('**/*.txt'):
        image_path = find_image(annotation_path)
        line_list = []
        with annotation_path.open('r', encoding='utf-8') as rf:
            for line in rf.readlines():
                data = line.split()
                if len(data) == 6:
                    class_id = int(data[0])
                    class_set.add(class_id)
                    if class_id == 0:
                        line_cnt += 1
                        line_list.append(line)
                elif '<!DOCTYPE html>' in line:
                    image_path.unlink()
                    annotation_path.unlink()
                    print(f'delete: {annotation_path}')
                    break
                else:
                    print(f'err: {annotation_path}')
                    break
        if len(line_list) == 0:
            print(f'empty: {annotation_path}')
            continue
        txt_cnt += 1
        out_annotation_path = make_out_path(annotation_path, data_root, out_path)
        with out_annotation_path.open('w', encoding='utf-8') as wf:
            wf.writelines(line_list)
        out_image_path = out_annotation_path.with_suffix(image_path.suffix)
        out_image_path.write_bytes(image_path.read_bytes())
    print(f'class_set: {class_set}, txt_cnt: {txt_cnt}, line_cnt: {line_cnt}')

if __name__ == "__main__":
    main()
