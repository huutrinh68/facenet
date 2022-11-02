import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import torch
import cv2
import json
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from torchvision import transforms as transforms


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
# mtcnn = MTCNN(image_size=160, margin=10)
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
              thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
              device=device, keep_all=True
              )
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# initialize deepsort
deepsort = DeepSort('./checkpoint/ckpt.t7',
                    max_dist=0.2, min_confidence=0.3,
                    max_iou_distance=0.7,
                    max_age=70, n_init=3, nn_budget=100,
                    use_cuda=True)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def create_database(databank, update=False):
    database = dict()
    database['database_path'] = databank
    database_list = list()

    if update:
        names = []
        assert os.path.exists('./tests/database.json'), 'database not existing'
        with open('./tests/database.json') as f:
            database = json.load(f)
        for obj in database['database']:
            names.append(obj['name'])

        database_list = database['database']

        for name in os.listdir(databank):
            # skip  old data
            if name in names:
                continue

            if not os.path.isdir(os.path.join(databank, name)):
                print('skip {}'.format(name))
                continue

            obj = dict()
            attribute_list = list()
            print(os.path.join(databank, name))
            for image_id in os.listdir(os.path.join(databank, name)):
                if not os.path.isfile(os.path.join(databank, name, image_id)):
                    print('skip {}'.format(image_id))
                    continue
                attribute = dict()

                attribute['path'] = os.path.join(databank, name, image_id)
                image = Image.open(os.path.join(databank, name, image_id))
                image_cropped = mtcnn(image)
                if image_cropped is not None:
                    embedding = resnet(image_cropped)
                    # convert tensor to list
                    embedding = embedding.detach().cpu().flatten().tolist()
                    attribute['emb'] = embedding
                    attribute_list.append(attribute)

            print(f'number of image {len(attribute_list)}')
            if len(attribute_list):
                obj['name'] = name
                obj['attribute'] = attribute_list
                obj['count'] = 0
                database_list.append(obj)
        database['database'] = database_list

        with open('./tests/database.json', 'w') as f:
            json.dump(database, f, indent=4)
        print('created database and saved')

    else:
        for name in os.listdir(databank):
            if not os.path.isdir(os.path.join(databank, name)):
                print('skip {}'.format(name))
                continue

            obj = dict()
            attribute_list = list()
            print(os.path.join(databank, name))
            for image_id in os.listdir(os.path.join(databank, name)):
                if not os.path.isfile(os.path.join(databank, name, image_id)):
                    print('skip {}'.format(image_id))
                    continue
                attribute = dict()

                attribute['path'] = os.path.join(databank, name, image_id)
                image = Image.open(os.path.join(databank, name, image_id))
                image_cropped = mtcnn(image)
                if image_cropped is not None:
                    embedding = resnet(image_cropped)
                    # convert tensor to list
                    embedding = embedding.detach().cpu().flatten().tolist()
                    attribute['emb'] = embedding
                    attribute_list.append(attribute)

            print(f'number of image {len(attribute_list)}')
            if len(attribute_list):
                obj['name'] = name
                obj['attribute'] = attribute_list
                obj['count'] = 0
                database_list.append(obj)
        database['database'] = database_list

        with open('./tests/database.json', 'w') as f:
            json.dump(database, f, indent=4)
        print('created database and saved')

    return database


def cos_similarity(p1, p2):
    return np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))


def recreate_embedding(embeddings, new_embedding):
    new_embedding = new_embedding.detach().cpu().flatten()
    embeddings.append(new_embedding)
    return embeddings


def main():
    cosin = torch.nn.CosineSimilarity(dim=2, eps=1e-6)

    # load if exist
    databank = './tests/databank'
    if os.path.exists('./tests/database.json'):
        with open('./tests/database.json') as f:
            database = json.load(f)
        print('load database')
    else:
        os.makedirs(databank, exist_ok=True)
        database = create_database(databank)
        print('create database')

    database_list = database.get('database', [])
    names = list()
    embeddings = list()

    for obj in database_list:
        name = obj.get('name', None)
        attribute = obj.get('attribute', [])

        if name is None or not len(attribute):
            continue
        for attr in attribute:
            embedding = attr.get('emb', None)
            if embedding is not None:
                names.append(name)
                embeddings.append(embedding)

    # image_path1 = "./databank/employee/1.jpg"
    # image_path2 = "./databank/obama/1.jpg"
    # image_path3 = "./databank/trump/1.jpg"
    # dataset = {'employee': image_path1, 'obama': image_path2, 'trump': image_path3}
    # names = list(dataset.keys())

    # img1 = Image.open(image_path1)
    # img_cropped1 = mtcnn(img1)
    # img_embedding1 = resnet(img_cropped1.unsqueeze(0))

    # img2 = Image.open(image_path2)
    # img_cropped2 = mtcnn(img2)
    # img_embedding2 = resnet(img_cropped2.unsqueeze(0))

    # img3 = Image.open(image_path3)
    # img_cropped3 = mtcnn(img3)
    # img_embedding3 = resnet(img_cropped3.unsqueeze(0))

    # embeddings = []
    # embeddings.append(img_embedding1.detach().cpu().flatten())
    # embeddings.append(img_embedding2.detach().cpu().flatten())
    # embeddings.append(img_embedding3.detach().cpu().flatten())

    # video_path = './tests/653205971.164770.mp4'
    # video_path = './tests/653205971.420371.mp4'
    # video_path = './tests/653205971.582914.mp4'
    # video_path = './tests/653205971.717960.mp4'
    video_path = './tests/obama_trump.mp4'
    # video_path = './tests/production ID_5198159.mp4'
    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('output_5.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    # # confustion matrix
    # dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
    # print(dists)

    # # distance to all
    # dists = [(img_embedding3 - e2).norm().item() for e2 in embeddings]
    # print(dists)

    # # matching
    # p1 = img_embedding1.squeeze().to('cpu').detach().numpy().copy()
    # p2 = img_embedding2.squeeze().to('cpu').detach().numpy().copy()
    # p3 = img_embedding3.squeeze().to('cpu').detach().numpy().copy()
    # img1vs2 = cos_similarity(p1, p2)
    # img1vs3 = cos_similarity(p1, p3)
    # print("1:2", img1vs2)
    # print("1:3", img1vs3)

    # assert len(embeddings) > 0, 'Please check database'
    if len(embeddings) > 0:
        print('embeddings has database')
        embeddings_ = np.stack(embeddings)
        embeddings_ = torch.from_numpy(embeddings_)

        # similirarity
        embeddings_ = embeddings_.unsqueeze(0)
        # sim = cosin(img_embedding1, embeddings)
        # print(sim.detach().numpy()[0])
    else:
        print('embeddings is empty')

    appear = []
    track_ids = []
    person_number = len(os.listdir(databank))
    while True:
        ok, origin_image = cap.read()
        if not ok:
            break

        image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
        output_image = origin_image.copy()
        image_cropped = mtcnn(image)
        if image_cropped is not None:
            face_locations, probs = mtcnn.detect(image)
            # face_embeddings = resnet(image_cropped.unsqueeze(0))
            face_embeddings = resnet(image_cropped)

            if face_locations is not None and len(face_locations):
                face_locations_ = xyxy2xywh(face_locations)
                # pass detections to deepsort
                # people class id is 0
                classes = torch.tensor([0 for _ in range(len(face_locations))])
                outputs = deepsort.update(face_locations_, probs, classes, origin_image)

                # draw boxes for visualization
                # if len(outputs) > 0:
                #     for j, (output, prob) in enumerate(zip(outputs, probs)):
                #         bboxes = output[0:4]
                #         id = output[4]
                #         label = f'{id}'
                #         # draw a box around the face
                #         x1, y1, x2, y2 = int(bboxes[0]), int(bboxes[1]), int(bboxes[2]), int(bboxes[3])
                #         cv2.rectangle(origin_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                #         # draw a label with a name below the face
                #         font = cv2.FONT_HERSHEY_DUPLEX
                #         cv2.putText(origin_image, label, (x1, y1 - 4), font, 0.75, (0, 255, 0), 1)

                for face_location, face_embedding, prob, output in zip(face_locations, face_embeddings, probs, outputs):
                    if prob < 0.95:
                        continue
                    # draw a box around the face
                    x1, y1, x2, y2 = int(face_location[0]), int(face_location[1]), int(face_location[2]), int(face_location[3])
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    id = output[4]

                    # best_match_index
                    if len(embeddings):
                        sim = cosin(face_embedding, embeddings_)
                        sim = (sim + 1) / 2
                        sim = sim.detach().numpy()[0]

                        best_match_index = np.argmax(sim)
                        name = names[best_match_index]

                        count_dict = dict()
                        for obj in database['database']:
                            obj_name = obj['name']
                            print(obj_name)
                            print(name)
                            print(appear)
                            if obj_name == name and name not in appear:
                                print('matching')
                                obj['count'] += 1
                                appear.append(name)

                            print('========')
                            # {'obama':1, 'trump':1, 'employee':2}
                            count_dict[obj_name] = obj['count']

                        # draw a label with a name below the face
                        font = cv2.FONT_HERSHEY_DUPLEX
                        # cv2.putText(output_image, names[best_match_index] + '_' + str(count_dict[name]) + '_' + str(id), (x1, y1 - 4), font, 0.75, (0, 255, 0), 1)
                        cv2.putText(output_image, names[best_match_index] + '_' + str(count_dict[name]), (x1, y1 - 4), font, 0.75, (0, 255, 0), 1)
                        # save image
                        best_sim = np.max(sim)
                    else:
                        best_sim = 0
                    # print(f'best similarity: {best_sim}')
                    if best_sim < 0.7 and id not in track_ids:
                        new_person = './tests/databank/p{0:07d}'.format(person_number)

                        image_num = 0
                        if os.path.isdir(new_person):
                            image_num = len(os.listdir(new_person))

                        print(f'number image of new_person: {image_num}')
                        # at least 2 image
                        if image_num < 2 and y2 - y1 > 30 or x2 - x1 > 30:
                            try:
                                save_image = origin_image[y1 - 20:y2 + 20, x1 - 20:x2 + 20]
                                print(f'save image in {new_person}/{image_num}.png')
                                # crop face and save
                                if 0 not in save_image.shape:
                                    os.makedirs(new_person, exist_ok=True)
                                    save_image = cv2.resize(save_image, (160, 160))
                                    cv2.imwrite(f'{new_person}/{image_num}.png', save_image)
                            except Exception as e:
                                print(e)

                        if os.path.isdir(new_person) and len(os.listdir(new_person)) > 1:
                            print(f'added track id {id}')
                            track_ids.append(id)
                            print(f'added person_{person_number} in to database')
                            names.append('p_{0:07d}'.format(person_number))
                            person_number += 1

                        # @TODO: need refine this
                        # save
                        if len(embeddings) > 0:
                            with open('./tests/database.json', 'w') as f:
                                json.dump(database, f, indent=4)
                        # update online
                        database = create_database(databank, update=True)
                        database_list = database.get('database', [])
                        names = list()
                        embeddings = list()
                        for obj in database_list:
                            name = obj.get('name', None)
                            attribute = obj.get('attribute', [])

                            if name is None or not len(attribute):
                                continue
                            for attr in attribute:
                                embedding = attr.get('emb', None)
                                if embedding is not None:
                                    names.append(name)
                                    embeddings.append(embedding)
                        if len(embeddings) > 0:
                            print('embeddings has database')
                            embeddings_ = np.stack(embeddings)
                            embeddings_ = torch.from_numpy(embeddings_)

                            # similirarity
                            embeddings_ = embeddings_.unsqueeze(0)

            else:
                deepsort.increment_ages()

        writer.write(output_image)
        cv2.imshow('Frame', output_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if len(embeddings) > 0:
        with open('./tests/database.json', 'w') as f:
            json.dump(database, f, indent=4)
    database = create_database(databank, update=True)
    with open('./tests/database.json', 'w') as f:
        json.dump(database, f, indent=4)

    writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
