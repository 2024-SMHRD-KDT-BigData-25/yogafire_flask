import os
import argparse
import time
import datetime
print(datetime.datetime.today(), "Evaluation start")

import numpy as np

from bpe import Config
from bpe.similarity_analyzer import SimilarityAnalyzer
from bpe.functional.motion import preprocess_motion2d_rc, cocopose2motion

def pad_to_16x(x):
    if x % 16 > 0:
        return x - x % 16 + 16
    return x


def pad_to_height(tar_height, img_height, img_width):
    scale = tar_height / img_height
    h = pad_to_16x(tar_height)
    w = pad_to_16x(int(img_width * scale))
    return h, w, scale


def preprocess_sequence(seq):
    for idx, seq_item in enumerate(seq):
        if len(seq_item) == 0:
            seq[idx] = seq[idx - 1]
        if idx > 0:
            seq_item[np.where(seq_item == 0)] = seq[idx - 1][np.where(seq_item == 0)]

    return seq

if __name__ == '__main__':
    t = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="sim_test", help="task name")
    parser.add_argument('--data_dir', default="./temp_data", required=False, help="path to dataset dir")
    parser.add_argument('--model_path', type=str, required=False,default="./train_log/exp_bpe/model/model_epoch2.pth")
    parser.add_argument('--video1', type=str, required=False, help="video1 mp4 path", default="no_video")
    parser.add_argument('--video2', type=str, required=False, help="video2 mp4 path", default="no_video")

    parser.add_argument('-v1', '--vid1_json_dir', type=str, required=False, help="video1's coco annotation json")
    parser.add_argument('-v2', '--vid2_json_dir', type=str, required=False, help="video2's coco annotation json")

    parser.add_argument('-h1', '--img1_height', type=int, help="video1's height", default=2500)
    parser.add_argument('-w1', '--img1_width', type=int, help="video1's width", default=2500)
    parser.add_argument('-h2', '--img2_height', type=int, help="video2's height", default=2500)
    parser.add_argument('-w2', '--img2_width', type=int, help="video2's width", default=2500)
    parser.add_argument('-pad2', '--pad2', type=int, help="video2's start frame padding", default=0)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)

    parser.add_argument('--output', type=str, required=False)
    parser.add_argument('--out_path', type=str, default='./visual_results', required=False)
    parser.add_argument('--out_filename', type=str, default='twice.mp4', required=False)
    parser.add_argument('--use_flipped_motion', action='store_true',
                        help="whether to use one decoder per one body part")
    parser.add_argument('--use_invisibility_aug', action='store_true',
                        help="change random joints' visibility to invisible during training")
    parser.add_argument('--debug', action='store_true', help="limit to 500 frames")
    # related to video processing
    parser.add_argument('--video_sampling_window_size', type=int, default=16,
                        help='window size to use for similarity prediction')
    parser.add_argument('--video_sampling_stride', type=int, default=16,
                        help='stride determining when to start next window of frames')
    parser.add_argument('--use_all_joints_on_each_bp', action='store_true',
                        help="using all joints on each body part as input, as opposed to particular body part")

    parser.add_argument('--similarity_measurement_window_size', type=int, default=1,
                        help='measuring similarity over # of oversampled video sequences')
    parser.add_argument('--similarity_distance_metric', choices=["cosine", "l2"], default="cosine")
    parser.add_argument('--privacy_on', action='store_true',
                        help='when on, no original video or sound in present in the output video')
    parser.add_argument('--thresh', type=float, default=0.5, help='threshold to seprate positive and negative classes')
    parser.add_argument('--connected_joints', action='store_true', help='connect joints with lines in the output video')

    args = parser.parse_args()
    # load meanpose and stdpose
    test_data_path = "D:/2022/NIA_80/BPE_Model/TEST_SET/필라테스"

    script_path = os.path.dirname(__file__)
    data_path = os.path.join(script_path,"data")
    #print(os.path.join(sript_path,"data", 'meanpose_rc_with_view_unit64.npy'))
    mean_pose_bpe = np.load(os.path.join(data_path, 'meanpose_rc_with_view_unit64.npy'))
    std_pose_bpe = np.load(os.path.join(data_path, 'stdpose_rc_with_view_unit64.npy'))

    #if not os.path.exists(args.out_path):
    #    os.makedirs(args.out_path)

    config = Config(args)
    model_path = os.path.join(data_path, "model_epoch2.pth")
    similarity_analyzer = SimilarityAnalyzer(config, model_path)

    # for NTU-RGB test - it used w:1920, h:1080
    h1, w1, scale1 = pad_to_height(config.img_size[0], args.img1_height, args.img1_width)
    h2, w2, scale2 = pad_to_height(config.img_size[0], args.img2_height, args.img2_width)

    from make_labelling import load_same_class_list, load_diff_class_list
    from sklearn.metrics import roc_auc_score
    same_class_list = load_same_class_list(test_data_path)
    diff_class_list = load_diff_class_list(test_data_path)
    true_score_list = []
    fw = open(os.path.join(os.path.dirname(test_data_path),"output.csv"),"w",encoding="utf8")
    print(os.path.join(os.path.dirname(test_data_path),"output.csv"))
    fw.write("labeling,sim_score,input_file_1, input_file_2\n")
    total_agg =  len(same_class_list) + len(diff_class_list)
    cnt = 0
    for (v1, v2) in same_class_list:
        # get input suitable for motion similarity analyzer
        seq1 = cocopose2motion(config.unique_nr_joints, v1, scale=scale1,
                               visibility=args.use_invisibility_aug)
        seq2 = cocopose2motion(config.unique_nr_joints, v2, scale=scale2,
                               visibility=args.use_invisibility_aug)[:, :, args.pad2:]

        # TODO: interpoloation or oef filtering for missing poses.
        seq1 = preprocess_sequence(seq1)
        seq2 = preprocess_sequence(seq2)

        seq1_origin = preprocess_motion2d_rc(seq1, mean_pose_bpe, std_pose_bpe,
                                             invisibility_augmentation=args.use_invisibility_aug,
                                             use_all_joints_on_each_bp=args.use_all_joints_on_each_bp)
        seq2_origin = preprocess_motion2d_rc(seq2, mean_pose_bpe, std_pose_bpe,
                                             invisibility_augmentation=args.use_invisibility_aug,
                                             use_all_joints_on_each_bp=args.use_all_joints_on_each_bp)

        # move input to device
        seq1_origin = seq1_origin.to(config.device)
        seq2_origin = seq2_origin.to(config.device)

        # get embeddings
        seq1_features = similarity_analyzer.get_embeddings(seq1_origin, video_window_size=args.video_sampling_window_size,
                                                           video_stride=args.video_sampling_stride)
        seq2_features = similarity_analyzer.get_embeddings(seq2_origin, video_window_size=args.video_sampling_window_size,
                                                           video_stride=args.video_sampling_stride)

        # get motion similarity
        motion_similarity_per_window = \
            similarity_analyzer.get_similarity_score(seq1_features, seq2_features,
                                                     similarity_window_size=args.similarity_measurement_window_size)
        if args.use_flipped_motion:
            seq1_flipped = preprocess_motion2d_rc(seq1, mean_pose_bpe, std_pose_bpe, flip=args.use_flipped_motion,
                                                  invisibility_augmentation=args.use_invisibility_aug,
                                                  use_all_joints_on_each_bp=args.use_all_joints_on_each_bp)
            seq1_flipped = seq1_flipped.to(config.device)
            seq1_flipped_features = similarity_analyzer.get_embeddings(seq1_flipped,
                                                                       video_window_size=args.video_sampling_window_size,
                                                                       video_stride=args.video_sampling_stride)
            motion_similarity_per_window_flipped = \
                similarity_analyzer.get_similarity_score(seq1_flipped_features, seq2_features,
                                                         similarity_window_size=args.similarity_measurement_window_size)
            for temporal_idx in range(len(motion_similarity_per_window)):
                for key in motion_similarity_per_window[temporal_idx].keys():
                    motion_similarity_per_window[temporal_idx][key] = max(motion_similarity_per_window[temporal_idx][key],
                                                                          motion_similarity_per_window_flipped[
                                                                              temporal_idx][key])

        # suppose same video horizontal
        video_width = int(config.img_size[0] / args.img1_height * args.img1_width + config.img_size[0] / args.img2_height * args.img2_width)
        video_height = config.img_size[0]
        #print("output:",)
        #print("out#put:", args.output)
        ra = 0
        la = 0
        rl = 0
        ll = 0
        torso = 0
        for num,  elt in enumerate(motion_similarity_per_window):
            ra += elt["ra"]
            la += elt["la"]
            rl += elt["rl"]
            ll += elt["ll"]
            torso += elt["torso"]
        score = abs(1/5 * (1/(num +1)) * (ra + la + rl + ll + torso))
        print(f"1,{score},{os.path.basename(v1)},{os.path.basename(v2)}", file=fw)
        print(f"1,{score},{os.path.basename(v1)},{os.path.basename(v2)}")
        fw.flush()


        cnt += 1
        #print(f"{cnt}/{total_agg},0,{score},{os.path.basename(v1)},{os.path.basename(v2)}")
        true_score_list.append(score)

    false_score_list = []
    for (v1, v2) in diff_class_list:
        # get input suitable for motion similarity analyzer
        seq1 = cocopose2motion(config.unique_nr_joints, v1, scale=scale1,
                               visibility=args.use_invisibility_aug)
        seq2 = cocopose2motion(config.unique_nr_joints, v2, scale=scale2,
                               visibility=args.use_invisibility_aug)[:, :, args.pad2:]

        # TODO: interpoloation or oef filtering for missing poses.
        seq1 = preprocess_sequence(seq1)
        seq2 = preprocess_sequence(seq2)

        seq1_origin = preprocess_motion2d_rc(seq1, mean_pose_bpe, std_pose_bpe,
                                             invisibility_augmentation=args.use_invisibility_aug,
                                             use_all_joints_on_each_bp=args.use_all_joints_on_each_bp)
        seq2_origin = preprocess_motion2d_rc(seq2, mean_pose_bpe, std_pose_bpe,
                                             invisibility_augmentation=args.use_invisibility_aug,
                                             use_all_joints_on_each_bp=args.use_all_joints_on_each_bp)

        # move input to device
        seq1_origin = seq1_origin.to(config.device)
        seq2_origin = seq2_origin.to(config.device)

        # get embeddings
        seq1_features = similarity_analyzer.get_embeddings(seq1_origin, video_window_size=args.video_sampling_window_size,
                                                           video_stride=args.video_sampling_stride)
        seq2_features = similarity_analyzer.get_embeddings(seq2_origin, video_window_size=args.video_sampling_window_size,
                                                           video_stride=args.video_sampling_stride)

        # get motion similarity
        motion_similarity_per_window = \
            similarity_analyzer.get_similarity_score(seq1_features, seq2_features,
                                                     similarity_window_size=args.similarity_measurement_window_size)
        if args.use_flipped_motion:
            seq1_flipped = preprocess_motion2d_rc(seq1, mean_pose_bpe, std_pose_bpe, flip=args.use_flipped_motion,
                                                  invisibility_augmentation=args.use_invisibility_aug,
                                                  use_all_joints_on_each_bp=args.use_all_joints_on_each_bp)
            seq1_flipped = seq1_flipped.to(config.device)
            seq1_flipped_features = similarity_analyzer.get_embeddings(seq1_flipped,
                                                                       video_window_size=args.video_sampling_window_size,
                                                                       video_stride=args.video_sampling_stride)
            motion_similarity_per_window_flipped = \
                similarity_analyzer.get_similarity_score(seq1_flipped_features, seq2_features,
                                                         similarity_window_size=args.similarity_measurement_window_size)
            for temporal_idx in range(len(motion_similarity_per_window)):
                for key in motion_similarity_per_window[temporal_idx].keys():
                    motion_similarity_per_window[temporal_idx][key] = max(motion_similarity_per_window[temporal_idx][key],
                                                                          motion_similarity_per_window_flipped[
                                                                              temporal_idx][key])

        # suppose same video horizontal
        video_width = int(config.img_size[0] / args.img1_height * args.img1_width + config.img_size[0] / args.img2_height * args.img2_width)
        video_height = config.img_size[0]
        #print("output:",)
        #print("out#put:", args.output)
        ra = 0
        la = 0
        rl = 0
        ll = 0
        torso = 0
        for num,  elt in enumerate(motion_similarity_per_window):
            ra += elt["ra"]
            la += elt["la"]
            rl += elt["rl"]
            ll += elt["ll"]
            torso += elt["torso"]
        score = abs(1/5 * (1/(num +1)) * (ra + la + rl + ll + torso))
        #print("F",score)
        print(f"0,{score},{os.path.basename(v1)},{os.path.basename(v2)}", file=fw)
        print(f"0,{score},{os.path.basename(v1)},{os.path.basename(v2)}")
        cnt += 1
        #print(f"{cnt}/{total_agg},0,{score},{os.path.basename(v1)},{os.path.basename(v2)}")
        false_score_list.append(score)
    true_label_list = [1 for _ in range(len(true_score_list))]
    false_label_list = [0 for _ in range(len(false_score_list))]

    fw.close()
    print("AUROC:",roc_auc_score(true_label_list + false_label_list, true_score_list + false_score_list))
    print(datetime.datetime.today(), "Evaluation end")

