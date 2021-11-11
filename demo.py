import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import torch
import cv2
import json


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
# mtcnn = MTCNN(image_size=160, margin=10)
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
              thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
              device=device
              )
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def create_database(databank):
    database = dict()
    database['database_path'] = databank
    database_list = list()

    for name in os.path.listdir(databank):
        if not os.path.isdir(name):
            continue

        obj = dict()
        attribute_list = list()
        for image_id in os.path.listdir(os.path.join(databank, name)):
            if not os.path.isfile(image):
                continue
            attribute = dict()

            attribute['path'] = os.path.join(databank, name, image_id)
            image = Image.open(os.path.join(databank, name, image_id))
            image_cropped = mtcnn(image)
            embedding = resnet(image_cropped.unsqueeze(0))
            embedding = embedding.detach().cpu().flatten()
            attribute['emb'] = embedding
            attribute_list.append(attribute)

        if len(attribute_list):
            obj['name'] = name
            obj['atttribute'] = attribute_list
            obj['count'] = 1
    database['database'] = database_list

    with open('./tests/database.json', 'w') as f:
        json.dump(database, f, indent=4)

    return database


def cos_similarity(p1, p2):
    return np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))


def main():
    cosin = torch.nn.CosineSimilarity(dim=2, eps=1e-6)

    # load if exist
    if os.path.exists('./tests/database.json'):
        database = json.load('./tests/database.json')
    else:
        database = create_database()

    database_list = database.get('database', [])
    names = list()
    embeddings = list()
    for obj in database_list:
        name = obj.get('name', None)
        attribute = obj.get('attribute', [])
        count = obj.get('count', 1)

        if name is not None and len(attribute):
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

    embeddings = np.stack(embeddings)
    embeddings = torch.from_numpy(embeddings)

    # video_path = './tests/653205971.164770.mp4'
    # video_path = './tests/653205971.420371.mp4'
    # video_path = './tests/653205971.582914.mp4'
    # video_path = './tests/653205971.717960.mp4'
    video_path = './tests/obama_trump.mp4'
    # databank = './tests/databank'
    cap = cv2.VideoCapture(video_path)
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

    # similirarity
    embeddings = embeddings.unsqueeze(0)
    # sim = cosin(img_embedding1, embeddings)
    # print(sim.detach().numpy()[0])
    while True:
        ok, origin_image = cap.read()
        if not ok:
            break

        image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
        image_cropped = mtcnn(image)
        face_locations, probs = mtcnn.detect(image)
        if image_cropped is not None:
            face_embeddings = resnet(image_cropped.unsqueeze(0))

            for face_location, face_embedding, prob in zip(face_locations, face_embeddings, probs):
                if prob < 0.85:
                    continue
                # best_match_index
                sim = cosin(face_embedding, embeddings)
                sim = (sim + 1) / 2
                sim = sim.detach().numpy()[0]
                best_match_index = np.argmax(sim)

                # display the results
                # draw a box around the face
                x, y, width, height = int(face_location[0]), int(face_location[1]), int(face_location[2]), int(face_location[3])
                cv2.rectangle(origin_image, (x, y), (width, height), (0, 0, 255), 2)
                # Draw a label with a name below the face
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(origin_image, names[best_match_index], (x, y - 4), font, 0.75, (0, 255, 0), 1)

        writer.write(origin_image)
        # cv2.imshow('Frame', origin_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
