import math
import time

DEBUG = False
file_setting_list = [  # item_file, training_file, testing_file
    ("ml-100k/u.item", "ml-100k/u.data", ""),
    ("ml-100k/u.item", "ml-100k/u1.base", "ml-100k/u1.test"),
    ("ml-100k/u.item", "ml-100k/u2.base", "ml-100k/u2.test"),
    ("ml-100k/u.item", "ml-100k/u3.base", "ml-100k/u3.test"),
    ("ml-100k/u.item", "ml-100k/u4.base", "ml-100k/u4.test"),
    ("ml-100k/u.item", "ml-100k/u5.base", "ml-100k/u5.test"),
]



class RecSys:
    def __init__(self, item_file, train_file, sim_function, test_file=None, is_print_log_info=False):
        self.is_print_log_info = is_print_log_info

        self.item_file = item_file
        self.train_file = train_file
        self.test_file = test_file
        self.sim_function = sim_function
        self.neigbour_num = None  # parameter for predicting method
        self.read_data()

        if self.is_print_log_info:
            print("Successfully initialized a recommendation system enviroment!!")
            UI_display_env(self)

    def set_neigbour_num(self, n):
        self.neigbour_num = n

    def read_data(self):
        # {id1: name1, id2: name2,...}

        if self.is_print_log_info:
            print("=====reading & processing item data...")
        self.item_dic = {}
        for line in open(self.item_file):
            if line.strip():
                id, name = line.split('|')[0:2]
                self.item_dic[id] = name

        if self.is_print_log_info:
            print("=====reading & processing training data...")
        self.train_data = {}
        for line in open(self.train_file):
            (user, item_id, rating, time) = line.split('\t')
            self.train_data.setdefault(user, {})
            self.train_data[user][item_id] = float(rating)

        if self.test_file is not None and self.test_file != "":
            if self.is_print_log_info:
                print("=====reading & processing testing data...")

            self.test_data = {}
            for line in open(self.test_file):
                (user_id, item_id, rating, time) = line.split('\t')
                self.test_data[(user_id, item_id)] = float(rating)

        # generate simularity data
        self.gen_sim_data()

    def gen_sim_data(self):
        if self.is_print_log_info:
            print("=====computing & generating similarity data between items...")
        t_train_data = self._transform_data(self.train_data)  # {item1:{user1:rating, ...}, ...}
        list_item = list(t_train_data)
        list_item.sort()
        self.sim_data = {}
        for i in range(len(list_item)):
            for j in range(i + 1, len(list_item)):
                item1 = list_item[i]
                item2 = list_item[j]
                # print(item1, item2)
                sim_val = self.sim_function(t_train_data, item1, item2)
                self.sim_data[(item1, item2)] = sim_val

    def _transform_data(self, data):
        result = {}
        for person in data:
            for item in data[person]:
                result.setdefault(item, {})

                result[item][person] = data[person][item]
        return result

    def get_user_rating(self, user_id, is_print=True):

        user_rating_record = self.train_data[user_id]

        if is_print:
            for item_id in user_rating_record:
                item_name = self.item_dic[item_id]
                rating = user_rating_record[item_id]
                print("{:s}     {:s}     ,rating: {:f}".format(item_id, item_name, rating))

        return user_rating_record

    def get_predict_rating(self, user_id, item_id):
        user_item_rating_dic = self.train_data[user_id]
        if item_id in user_item_rating_dic.keys():
            return user_item_rating_dic[item_id]

        score_for_the_item = 0.0
        total_sim = 0.0

        # TODO: test
        # test_dic = {}
        # for i in user_item_rating_dic:
        #     test_dic[self.item_dic[i]]=user_item_rating_dic[i]
        #
        # test_list = list(test_dic.keys())
        # test_list.sort()
        # for k in test_list:
        #     print(k + ":" + str(test_dic[k]))


        # find all items in user's rated item data, get user rating, and similarity to target_item.
        # store to tmp_sim_rating_pair
        tmp_sim_rating_pair = []  # [(sim_val, rating), ...]
        for (user_rated_item, rating) in user_item_rating_dic.items():

            item_pair = (user_rated_item, item_id)
            if (item_pair not in self.sim_data):
                item_pair = (item_id, user_rated_item)

            if (item_pair not in self.sim_data):
                continue

            sim_val = self.sim_data[item_pair]
            tmp_sim_rating_pair.append((sim_val, rating))

        # trim, find <neigbour_num> pair of (sim_val, rating) with highest sim_val in tmp_sim_rating_pair
        tmp_sim_rating_pair.sort(reverse=True)
        if self.neigbour_num != None:
            tmp_sim_rating_pair = tmp_sim_rating_pair[0:self.neigbour_num]
        for (sim_val, rating) in tmp_sim_rating_pair:
            score_for_the_item += sim_val * rating
            total_sim += sim_val

        if total_sim == 0.0:
            return 0.0
        return score_for_the_item / total_sim

    def get_recommendation_list(self, user_id, top_n=None, is_print=True):
        set_user_rated_item = set(self.train_data[user_id].keys())
        set_all_item = set(self.item_dic.keys())
        set_unrated_item = set_all_item - set_user_rated_item

        list_predicted_rating = []  # [(rating, item_id),...]
        for item_id in set_unrated_item:
            p_rating = self.get_predict_rating(user_id, item_id)
            list_predicted_rating.append((p_rating, item_id))

        list_predicted_rating.sort(reverse=True)

        if top_n != None:
            list_predicted_rating = list_predicted_rating[0:top_n]

        if is_print == True:
            print("===================")
            print("Recommendation List")
            print("===================")

            top_num = 1
            for pair in list_predicted_rating:
                rating, item_id = pair
                print("Top{:d}: {:s}, rating: {:.4f}".format(top_num, self.item_dic[item_id], rating))

                top_num += 1

        return list_predicted_rating

    def eval_prediction(self):
        if not hasattr(self, 'test_data') or not self.test_data:
            print("No testing data")
            return

        list_abs_diff = []
        list_square_diff = []
        count_significant_diff = 0

        for key in self.test_data:
            real_rating = self.test_data[key]
            predict_rating = self.get_predict_rating(key[0], key[1])

            # ignore unpredictable condition: new user or new item
            if predict_rating == 0:
                continue

            abs_diff = abs(predict_rating - real_rating)
            if abs_diff > 1:
                count_significant_diff += 1

            square_diff = (predict_rating - real_rating) ** 2
            list_abs_diff.append(abs_diff)
            list_square_diff.append(square_diff)

        MAE = sum(list_abs_diff) / len(list_abs_diff)
        RMSE = (sum(list_square_diff) / len(list_square_diff)) ** (1 / 2)
        print("MAE={:.4f}, RMSE={:.4f}".format(MAE, RMSE))
        print("significant err prediction number is: {:d}".format(count_significant_diff))
        return (MAE, RMSE)


# ========================END of Class define================================

# Euclidean distance score algorithm
def euclidean_distance(data, item1, item2):
    shared_item = {}
    for item in data[item1]:
        if item in data[item2]:
            shared_item[item] = 1
    if len(shared_item) == 0: return 0

    sum_of_squares = sum([pow(data[item1][item] - data[item2][item], 2)
                          for item in data[item1] if item in data[item2]])
    return 1 / (1 + math.sqrt(sum_of_squares))


# Pearson correlation score algorithm
def pearson_distance(data, item1, item2):
    shared_item = {}
    for item in data[item1]:
        if item in data[item2]:
            shared_item[item] = 1

    length = len(shared_item)
    if length == 0: return 0

    sum1 = sum(data[item1][item] for item in shared_item)
    sum2 = sum(data[item2][item] for item in shared_item)

    sum1_of_squares = sum([pow(data[item1][item], 2) for item in shared_item])
    sum2_of_squares = sum([pow(data[item2][item], 2) for item in shared_item])

    sum_of_cross = sum([data[item1][item] * data[item2][item] for item in shared_item])

    num = sum_of_cross - (sum1 * sum2 / length)
    den = math.sqrt((sum1_of_squares - pow(sum1, 2) / length) * (sum2_of_squares - pow(sum2, 2) / length))

    if den == 0: return 0
    result = num / den

    if result > 1:
        result = 1.0
    if result < -1:
        result = -1.0
    result = round((result + 1) / 2, 4)
    return result


def UI():
    # choose file setting
    print("Choose one setting of file to init the environment:")
    print("{:5s}  {:20s}    {:20s}    {:20s}".format("ID", "item file", "training file", "testing file"))

    for id, file_pair in enumerate(file_setting_list):
        print("{:<5d}  {:20s}    {:20s}    {:20s}".format(id, file_pair[0], file_pair[1], file_pair[2]))

    file_setting_ID = int(input("Type the ID of Setting: "))

    FILE_SETTING = file_setting_list[file_setting_ID]

    # choose simularity function
    print("")
    print("Choose one simularity function:")
    print("ID=0    use euclidean distance")
    print("ID=1    use pearson distance")

    while True:
        sim_function_ID = int(input("Type the ID of Setting: "))

        if sim_function_ID == 0:
            sim_function = euclidean_distance
            break
        elif sim_function_ID == 1:
            sim_function = pearson_distance
            break
        else:
            print("wrong ID!")

    # init system environment
    print("")
    print("")
    test_sys = RecSys(item_file=FILE_SETTING[0], train_file=FILE_SETTING[1], test_file=FILE_SETTING[2],
                      sim_function=pearson_distance, is_print_log_info=True)

    # operating
    op_dic = {
        '1': "Display user's rated movies",
        '2': "Predict user's rating on a movie",
        '3': "Display our recommendation list for user",
        '4': "Evaluate the recommendation system1: Prediction Evaluation",
        '5': "Evaluate the recommendation system2: Recommendation List Evaluation",
        '6': "Setup Neigbour Num (paramter for prediction. now it is: {:s} )".format(str(test_sys.neigbour_num)),
        '7': "Display System Environment",
        '0': "Exit"
    }

    while True:
        print("")
        print("")

        op_id = input("Type the operation ID: (type ? for menu)")
        if op_id == '?':
            print("Operation Menu:")
            for id in sorted(op_dic.keys()):
                print("ID={:s}    {:s}".format(id, op_dic[id]))
            print("You can find user id & item id in user file & item file!")
        elif op_id == '1':
            UI_user_rated_items(test_sys)
        elif op_id == '2':
            UI_predict(test_sys)
        elif op_id == '3':
            UI_rec_list(test_sys)
        elif op_id == '4':
            UI_eval_prediction(test_sys)
        elif op_id == '5':
            UI_eval_rec_list(test_sys)
        elif op_id == '6':
            UI_setup_nigbour_num(test_sys)
        elif op_id == '7':
            UI_display_env(test_sys)
        elif op_id == '0':
            break
        else:
            print("Wrong operation ID")

    print("Exited.")


def UI_user_rated_items(test_sys: RecSys):
    print("")
    user_id = input("Type User ID: ")
    test_sys.get_user_rating(user_id=user_id, is_print=True)


def UI_predict(test_sys: RecSys):
    print("")
    user_id = input("Type User ID: ")
    time.sleep(0.5)
    item_id = input("Type Movie ID: ")
    predict_rating = test_sys.get_predict_rating(user_id=user_id, item_id=item_id)
    print("=====predict result=====")
    print(predict_rating)


def UI_rec_list(test_sys: RecSys):
    print("")
    user_id = input("Type User ID: ")
    time.sleep(0.5)
    top_n = int(input("Type How many recommendation movies you want to get: "))
    test_sys.get_recommendation_list(user_id=user_id, top_n=top_n, is_print=True)


def UI_eval_prediction(test_sys: RecSys):
    print("")
    UI_display_env(test_sys)
    print("=======================Result===================================")
    test_sys.eval_prediction()


# TODO:...
def UI_eval_rec_list(test_sys: RecSys):
    print("UI In construction...")
    pass


def UI_setup_nigbour_num(test_sys: RecSys):
    print("")
    print("a parameter that will effect predicting! Be careful!")
    neigbour_num = input("Type Number of neigbour you want: ")
    time.sleep(0.5)
    confirm = input("Confirm?(Y/N)")
    if confirm == "Y":
        test_sys.neigbour_num = neigbour_num
        print("setup succeed!")
    else:
        print("not work!")


def UI_display_env(test_sys: RecSys):
    print("")
    print("System Environment:")
    print("item file: {:s},    training file: {:s}    test file: {:s}".format(test_sys.item_file, test_sys.train_file,
                                                                              test_sys.test_file))
    print("choosed similarity computing function: {:s}".format(str(test_sys.sim_function)))
    print("Number of neigbour={:s}".format(str(test_sys.neigbour_num)))


def my_test1():
    file_setting = file_setting_list[1]
    test1 = RecSys(item_file=file_setting[0], train_file=file_setting[1], test_file=file_setting[2],
                   sim_function=pearson_distance)
    for n in (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100):
        print("neigbour_num={:d}, ".format(n), end="")
        test1.neigbour_num = n
        test1.eval_prediction()


def my_test2():
    for file_pair in file_setting_list[1:]:
        print(file_pair)
        test1 = RecSys(item_file=file_pair[0], train_file=file_pair[1], test_file=file_pair[2],
                       sim_function=euclidean_distance)
        test1.set_neigbour_num(60)
        test1.eval_prediction()


if __name__ == '__main__':
    UI()

    # my_test2()

    # test1.get_user_rating("87")
    # test1.get_recommendation_list(user_id="87", is_print=True)

    # for key in test1.sim_data:
    #     print("{:s}|{:s}|{:.4f}".format(key[0],key[1],test1.sim_data[key]))

    # print(test1.get_predict_rating(user_id="87",item_id="1"))
    # test1.get_recommendation_list(user_id="87",is_print=True)

    print("END")
    exit(0)
