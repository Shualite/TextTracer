import json
import numpy as np
import os
import editdistance as ed
import math
import re

def calc_iou(box_i, box_j):
    left_i, right_i, top_i, bottom_i = np.min(box_i[::2]), np.max(box_i[::2]), np.min(box_i[1::2]), np.max(box_i[1::2])
    left_j, right_j, top_j, bottom_j = np.min(box_j[::2]), np.max(box_j[::2]), np.min(box_j[1::2]), np.max(box_j[1::2])
    area_i = (bottom_i-top_i) * (right_i-left_i)
    area_j = (bottom_j-top_j) * (right_j-left_j)
    union_lt_x, union_lt_y, union_rb_x, union_rb_y = np.maximum(left_i, left_j), np.maximum(top_i, top_j), np.minimum(right_i, right_j), np.minimum(bottom_i, bottom_j)
    union_w, union_h = np.maximum(0, union_rb_x-union_lt_x), np.maximum(0, union_rb_y-union_lt_y)
    union_areas = union_w * union_h
    # normal iou
    iou = union_areas/(area_i+area_j-union_areas)
    return iou

def filter_number_symbol(line):
    crop = re.compile("[0-9* ]")
    result = crop.sub('', line)
    return result


class TextTracer:
    def __init__(self, ):
        self._init_config()
    
    def _init_config(self, ):
        # 从视频中间部分随机选帧
        self.video_between = (0.1, 0.9)
        
        # 帧间IOU阈值，用于bbox绑定
        self.frames_iou_thresh = 0.6
        # 章节标题字数上限
        self.max_title_len = 32
        # 文本的编辑距离，容错百分比
        self.ed_error_rate = 0.4
        
        self.center_region_between = (0.3, 0.7)
        self.group_center_between = (0.375, 0.625)
        self.group_line_width_thresh = 0.68
        # self.group_line_avg_h_min = 0.026
        self.group_line_avg_h_min = 0.024
        self.group_line_avg_h_max = 0.050
        
        # self.group_line_h_std_thresh = 0.004
        self.group_line_h_std_thresh = 0.003
        # 允许中心点抖动范围，按行聚簇
        self.cluster_thresh = 0.01
        # 要求出现频率
        self.high_freq_rate = 0.7
        self.center_x_gap_thresh=0.6
        
        # 小标题至少有三个及以上
        self.section_number_thresh = 3
        # 至少有两个 内容有效的字段
        self.section_valid_number_thresh = 2
        
        self.frame_num = -1
        self.min_frames_num = 1
    
    def _init_buffer(self,):
        self.buffer_boxes_pre = []
        self.buffer_boxes_cur = []
        self.buffer_results_pre = []
        self.buffer_results_cur = []
    
    def candidatefinder_for_idx(self, query_box):
        if len(self.buffer_boxes_pre) == 0:
            return np.array([], dtype=np.int64)

        query_left, query_right, query_top, query_bottom = min(query_box[::2]), max(query_box[::2]), min(query_box[1::2]), max(query_box[1::2])
        # 获取前一帧的bbox信息
        left, right, top, bottom = np.min(self.buffer_boxes_pre[:, ::2], axis=1), np.max(self.buffer_boxes_pre[:, ::2], axis=1), np.min(self.buffer_boxes_pre[:, 1::2], axis=1), np.max(self.buffer_boxes_pre[:, 1::2], axis=1)
        
        query_h, query_w = (query_bottom-query_top), (query_right-query_left)
        query_area = query_h * query_w
        
        areas = (right-left)*(bottom-top)
        
        union_lt_x, union_lt_y, union_rb_x, union_rb_y = np.maximum(query_left, left), np.maximum(query_top, top), np.minimum(query_right, right), np.minimum(query_bottom, bottom)
        union_w, union_h = np.maximum(0, union_rb_x-union_lt_x), np.maximum(0, union_rb_y-union_lt_y)
        union_areas = union_w * union_h
        # normal iou
        ious = union_areas/(query_area+areas-union_areas)
        # single iou
        # iou_v1 = union_areas/np.maximum(query_area, union_areas)
        
        idx = np.where(ious>self.frames_iou_thresh)[0]
        
        return idx
    
    def det_recog_bind(self, frame_id, cur_frame_result):
        for box_id, item in enumerate(cur_frame_result):
            if 'location' not in item.keys() or 'text' not in item.keys() or 'colconfidence' not in item.keys():
                continue
            
            query_box = item['location'].reshape(-1)
            
            item['location'] = [query_box]
            item['frame_id'] = [str(frame_id)]
            item['text'] = [item['text']]
            item['colconfidence'] = [item['colconfidence']]

            query_box = item['location'][0]
            
            query_text = item['text'][0]
            if len(query_text) <= self.max_title_len:
                # step 1. match previous frame results
                candidate_idx = self.candidatefinder_for_idx(query_box)
                # if match fail, then create new
                if not bool(candidate_idx.size):
                    self.buffer_results_cur.append(item)
                    self.buffer_boxes_cur.append(item['location'][0])
                    continue
                
                # considering match more than one instance.
                candidate_eds = []
                for idx in candidate_idx.tolist():
                    temp_item = self.buffer_results_pre[idx]

                    # step 2. compare text edit distance
                    optimal_text_idx = np.argmax(temp_item['colconfidence'])
                    temp_text = temp_item['text'][optimal_text_idx]
                    temp_ed = ed.eval(temp_text, query_text)
                    candidate_eds.append(temp_ed)
                # choosing max iou item, calc editdistance.
                best_ed_idx = np.argsort(candidate_eds)[0]
                text_ed = candidate_eds[best_ed_idx]

                # step 3. merge or create according to text_ed
                max_ed = max(1, int(len(query_text)*self.ed_error_rate))
                if text_ed <= max(1, max_ed):
                    candidate_item = self.buffer_results_pre[candidate_idx[best_ed_idx]]
                    candidate_item['location'].append(item['location'][0])
                    candidate_item['text'].append(item['text'][0])
                    candidate_item['colconfidence'].append(item['colconfidence'][0])
                    candidate_item['frame_id'].append(item['frame_id'][0])
                    
                    self.buffer_results_cur.append(candidate_item)
                    self.buffer_boxes_cur.append(candidate_item['location'][0])
                else:
                    self.buffer_results_cur.append(item)
                    self.buffer_boxes_cur.append(item['location'][0])
            else:
                pass
    
    def info_extract(self, detailed_results):
        items = []
        
        # step1. clean info for each item
        for item in detailed_results:
            texts, locations, scores, frame_ids = item['text'], item['location'], item['colconfidence'], item['frame_id']
            max_score_id = np.array(scores).argmax()
            
            # TODO: check 是否合理要大于等于 N 帧
            if len(frame_ids) >= self.min_frames_num:
                items.append({
                    'text': texts[max_score_id],
                    'loc': locations[max_score_id],
                    'score': scores[max_score_id],
                    'selected_frame': frame_ids[max_score_id], 
                    'frame_ids': frame_ids,
                })
    
        # step2. remove redundent text between items
        item_num = len(items)
        deleted_idx = np.zeros((item_num,))
        for i in range(item_num):
            for j in range(item_num):
                if i != j and deleted_idx[i]==0 and deleted_idx[j]==0:
                    text_i, text_j = filter_number_symbol(items[i]['text']), \
                                    filter_number_symbol(items[j]['text'])
                    temp_ed = ed.eval(text_i, text_j)
                    ed_thresh = max(1, math.ceil(0.2 * (len(text_i)+len(text_j))/2))
                    
                    box_i, box_j = items[i]['loc'], items[j]['loc']
                    iou = calc_iou(box_i, box_j)
                        
                    if iou>self.frames_iou_thresh and temp_ed <= ed_thresh:
                        # 考虑 如何融合
                        score_i, score_j = items[i]['score'], items[j]['score']
                        idx_list = (i, j)
                        min_score_id = np.argmin([score_i, score_j])
                        max_score_id = np.argmax([score_i, score_j])
                        # merge min_score_id to max_score_id
                        deleted_idx[idx_list[min_score_id]] = 1
                        items[idx_list[max_score_id]]['frame_ids'].extend(items[idx_list[min_score_id]]['frame_ids'])
                        items[idx_list[max_score_id]]['frame_ids'] = sorted(items[idx_list[max_score_id]]['frame_ids'], key=lambda x: int(x))
                        
        selected_idx = np.where(deleted_idx==0)[0]
        results = np.array(items)[selected_idx]
        return results
    
    def __call__(self, data_dict):
        self._init_buffer()
        
        final_results = []
        for frame_id, frame_data in data_dict.items():
            self.det_recog_bind(frame_id, frame_data)
            
            new_items = []
            for i in self.buffer_results_cur:
                exist = False
                for item in self.buffer_results_pre:
                    if i['text'] == item['text'] and (np.array(i['location']) == np.array(item['location'])).all():
                        exist = True
                        break
                if not exist:
                    new_items.append(i)
            
            final_results.extend(new_items)
            # 完成一帧的匹配，交替缓存区
            self.buffer_boxes_pre = np.array(self.buffer_boxes_cur)
            self.buffer_results_pre = np.array(self.buffer_results_cur)
            self.buffer_boxes_cur = []            
            self.buffer_results_cur = []
            assert len(self.buffer_boxes_pre) == len(self.buffer_results_pre)
        
        results = self.info_extract(final_results)
        return results