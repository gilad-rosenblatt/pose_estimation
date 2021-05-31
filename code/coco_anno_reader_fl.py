import json


def get_input_for_fl(json_path, min_area=10000, max_area=100000, category_id=1):
    # gets image names with bbox. box area between (min_area-max_area), only from category_id (1==person)
    f = open(json_path)
    data = json.load(f)
    f.close()
    file_info = data['images']
    id_to_info = dict()

    for ind, file in enumerate(file_info):
        id_to_info[file['id']] = file

    anno = data['annotations']
    human_bbox_id = []
    for ind, an in enumerate(anno):
        if an['category_id'] == category_id and (min_area < an['area'] < max_area):
            human_bbox_id.append([an['image_id'], an['bbox']])

    id_to_bbox = dict()
    for bbox in human_bbox_id:
        key = bbox[0]
        if not key in id_to_bbox:
            id_to_bbox[key] = [bbox[1]]
        else:
            id_to_bbox[key].append(bbox[1])
    info = []
    for key in id_to_bbox.keys():
        info.append([id_to_info[key]['file_name'], id_to_bbox[key]])

    return info
