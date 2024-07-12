import pickle
import numpy as np
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/media/kaan/Extreme SSD/nuscenes/', help='dataset path')
    parser.add_argument('--classes', type=str, nargs='*', default=['car', 'truck', 'bus', 'trailer'], help='classes to use')
    return parser.parse_args()


def main():

    args = parse_arguments()

    db_info_path = args.dataset_path + "nuscenes_dbinfos_train.pkl"
    with open(db_info_path, 'rb') as f:
        data = pickle.load(f)

    print(data.keys())

    print("Dataset label sizes:")
    print("car: " + str(len(data['car'])))
    print("truck: " + str(len(data['truck'])))
    print("pedestrian: " + str(len(data['pedestrian'])))
    print("motorcycle: " + str(len(data['motorcycle'])))
    print("bicycle: " + str(len(data['bicycle'])))
    print("bus: " + str(len(data['bus'])))
    print("trailer: " + str(len(data['trailer'])))


    print(args.classes)

    #split the dataset
    split = 0.8
    train_data = {}
    test_data = {}
    for key in args.classes:
        data[key] = np.array(data[key])
        np.random.shuffle(data[key])
        split_idx = int(len(data[key]) * split)
        train_data[key] = data[key][:split_idx]
        test_data[key] = data[key][split_idx:]

    #save test and train
    db_info_path = args.dataset_path + "dbinfos_train.pkl"
    with open(db_info_path, 'wb') as f:
        pickle.dump(train_data, f)

    db_info_path = args.dataset_path + "dbinfos_val.pkl"
    with open(db_info_path, 'wb') as f:
        pickle.dump(test_data, f)

    print("Train data sizes:")
    print("car: " + str(len(train_data['car'])))
    print("truck: " + str(len(train_data['truck'])))
    print("bus: " + str(len(train_data['bus'])))
    print("trailer: " + str(len(train_data['trailer'])))

    print("Test data sizes:")
    print("car: " + str(len(test_data['car'])))
    print("truck: " + str(len(test_data['truck'])))
    print("bus: " + str(len(test_data['bus'])))
    print("trailer: " + str(len(test_data['trailer'])))
    print(test_data['car'][0].keys())
    print(test_data['car'][0])


    db_info_path = "/home/kaan/dataset_concat/dbinfos_val.pkl"
    with open(db_info_path, 'rb') as f:
        data = pickle.load(f)
    print(data['car'][0].keys())
    print(data['car'][0])

if __name__ == '__main__':
    main()
