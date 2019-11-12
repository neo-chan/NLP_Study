#coding=utf-8
import pandas as pd
import jieba
import jieba.posseg as pseg
train_data_path=r"../datasource/AutoMaster_TrainSet.csv"
test_data_path=r"../datasource/AutoMaster_TestSet.csv"
train_data_input=r"../datasource/train_data_input.txt"
train_data_output=r"../datasource/train_data_output.txt"
test_data_input=r"../datasource/test_data_input.txt"
test_data_output=r"../datasource/test_data_output.txt"
vocab_to_index=r"../datasource/vocab_to_index.txt"
index_to_vocab=r"../datasource/index_to_vocab.txt"
#train_data_path=r"/home/cxb/PycharmProjects/NLP_Study/datasource/AutoMaster_TrainSet.csv"
#test_data_path=r"/home/cxb/PycharmProjects/NLP_Study//datasource/AutoMaster_TestSet.csv"
#train_data_input_voc=r"/home/cxb/PycharmProjects/NLP_Study/datasource/train_data_input_voc.txt"
#train_data_output_voc=r"/home/cxb/PycharmProjects/NLP_Study/datasource/train_data_output_voc.txt"
#test_data_input_voc=r"/home/cxb/PycharmProjects/NLP_Study/datasource/test_data_input_voc.txt"
#test_data_output_voc=r"/home/cxb/PycharmProjects/NLP_Study/datasource/test_data_output_voc.txt"

def clear(comment):
    comment = comment.strip()
    comment = comment.replace('、', '')
    comment = comment.replace('，', '。')
    comment = comment.replace('《', '。')
    comment = comment.replace('》', '。')
    comment = comment.replace('～', '')
    comment = comment.replace('…', '')
    comment = comment.replace('\r', '')
    comment = comment.replace('\t', ' ')
    comment = comment.replace('\f', ' ')
    comment = comment.replace('/', '')
    comment = comment.replace('、', ' ')
    comment = comment.replace('/', '')
    comment = comment.replace('。', '')
    comment = comment.replace('（', '')
    comment = comment.replace('"', '')
    comment = comment.replace('“', '')
    comment = comment.replace('）', '')
    comment = comment.replace('_', '')
    comment = comment.replace('-', '')
    comment = comment.replace('?', ' ')
    comment = comment.replace('？', ' ')
    comment = comment.replace('|', '')
    comment = comment.replace('：', '')
    comment = comment.replace('！', '')
    comment = comment.replace('!', '')
    comment = comment.replace('[语音]', '')
    comment = comment.replace('[图片]', '')
    comment = comment.replace('技师说', '')
    comment = comment.replace('车主说', '')
    comment = comment.replace('吧', '')
    comment = comment.replace('了', '')
    comment = comment.replace('啊', '')
    comment = comment.replace('呢', '')
    return comment

def seg_cut(contents):
    ret=[]
    for each in contents:
        if isinstance(each,str):
            clear_words=clear(each)
            seg_list=jieba.cut(clear_words)
            ret.append(" ".join(seg_list))
    return ret

def del_incomplete_data_row(path):
    df=pd.read_csv(path,encoding="utf-8")
    df2=df.dropna()
    return df2

def parse_data(path):

    #df=pd.read_csv(path,encoding="utf-8")
    #df2=df.dropna()
    df2=del_incomplete_data_row(path)
    if 'Question' and 'Dialogue' in df2.columns:
        data_x = df2.Question.str.cat(df2.Dialogue)
    data_y=[]
    if 'Report' in df2.columns:
        data_y=df2.Report
    return data_x,data_y

def save_data(data,path,front_index=True):
    with open(path,"w",encoding="utf-8") as f:
        for i,seg in enumerate(data):
            if isinstance(seg,str):
                if front_index:
                    f.write("%s " %i)
                    f.write("%s" %seg)
                else:
                    f.write("%s " %seg)
                    f.write("%s" %i)
            f.write("\n")

def construct_vocab():
    train_df=del_incomplete_data_row(train_data_path)
    test_df=del_incomplete_data_row(test_data_path)
    train_data=train_df.Brand.str.cat(train_df.Model).str.cat(train_df.Question).str.cat(train_df.Dialogue).str.cat(train_df.Report)
    test_data=test_df.Brand.str.cat(test_df.Model).str.cat(test_df.Question).str.cat(test_df.Dialogue)
    allwords=set()
    train_words=seg_cut(train_data)
    test_words=seg_cut(test_data)
    for each in train_words:
        aa=each.split()
        for i in aa:
            if isinstance(i,str):
                allwords.add(i)
    for each in test_words:
        aa=each.split()
        for i in aa:
            if isinstance(i,str):
                allwords.add(i)
    save_data(allwords,vocab_to_index,False)
    save_data(allwords,index_to_vocab)

if __name__ == '__main__':
    construct_vocab()
    train_x,train_y=parse_data(train_data_path)
    test_x,test_y=parse_data(test_data_path)
    train_seg_x=seg_cut(train_x)
    train_seg_y=seg_cut(train_y)
    test_seg_x=seg_cut(test_x)
    test_seg_y=seg_cut(test_y)
    save_data(train_seg_x,train_data_input)
    save_data(train_seg_y,train_data_output)
    save_data(test_seg_x,test_data_input)
    save_data(test_seg_y,test_data_output)




