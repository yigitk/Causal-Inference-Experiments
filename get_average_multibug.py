import sys
import os

version_bug_dict = {
    "J0Long2Bug": ["d_1", "seps_0"],
    "J0Long3Bug": ["d_1", "seps_0", "d_4"],
    "lngammaLong2Bug": ["x_9", "result_val_41"],
    "lngammaLong3Bug": ["x_9", "result_val_41", "result_err_61"],
    "matric_transposeLong2Bug": ["m_data_e1_1", "m_data_e2_1"],
    "matric_transposeLong3Bug": ["m_data_e1_1", "m_data_e2_1", "tmp_0"],
    "synchrotron_1Long2Bug": ["x_4", "t_1"],
    "synchrotron_1Long3Bug": ["x_4", "t_1", "y_0"]
}

version_target_vars_dict = {
    "J0Long2Bug": 40,
    "J0Long3Bug": 40,
    "lngammaLong2Bug": 74,
    "lngammaLong3Bug": 74,
    "matric_transposeLong2Bug": 24,
    "matric_transposeLong3Bug": 24,
    "synchrotron_1Long2Bug": 37,
    "synchrotron_1Long3Bug": 37,
}

results_dict = {}

directory = os.fsencode(".")

for program in os.listdir(directory):
    if not os.path.isdir(program):
        continue
    for version in os.listdir(os.path.join(directory, program)):
        if "2Bug" not in str(version) and "3Bug" not in str(version):
            continue
        bugs = version_bug_dict[str(version)[2:-1]]
        target_vars = version_target_vars_dict[str(version)[2:-1]]
        for method in os.listdir(os.path.join(directory, program, version)):
            for perc in os.listdir(os.path.join(directory, program, version, method)):
                ranks = []
                for trial in os.listdir(os.path.join(directory, program, version, method, perc)):
                    filename = trial
                    all_file_ranks = []
                    with open(os.path.join(directory, program, version, method, perc, filename)) as f:
                        for _ in range(4):
                            next(f)
                        line_number = 0
                        bug_scores = []
                        for line in f:
                            line_number += 1
                            rank = line_number - 4
                            variable = line.split(",")[1]
                            score = line.split(",")[2]
                            for bug in bugs:
                                if variable == bug:
                                    bug_scores.append((bug, float(score)))
                            all_file_ranks.append((variable, float(score)))
                        top_bug = bug_scores[0]
                        for bug_score_tup in bug_scores:
                            top_score = 0
                            print(bug_score_tup[1], top_score)
                            if bug_score_tup[1] > top_score:
                                top_bug = bug_score_tup
                                top_score = bug_score_tup[0]
                        print(bug_scores)
                        print("first found", top_bug[0])
                        ties = []
                        for i, tup in enumerate(all_file_ranks):
                            if tup[1] == top_bug[1]:
                                ties.append(i)
                        rank = sum(ties) / len(ties) + 1
                        ranks.append(rank)
                average_rank = sum(ranks) / len(ranks)
                method_str = str(method)[2:-1]
                program_name = str(version)[2:-1]
                perc_str = str(perc)[2:-1]
                base = (str(version)[2:-1], perc_str,
                        str("\"" + ",".join(bugs) + "\""), target_vars)

                if base not in results_dict:
                    results_dict[base] = [0, 0]
                    if method_str == "FL":
                        results_dict[base][0] = average_rank
                    if method_str == "ESP":
                        results_dict[base][1] = average_rank
                else:
                    if method_str == "FL":
                        results_dict[base][0] = average_rank
                    if method_str == "ESP":
                        results_dict[base][1] = average_rank
for base in results_dict:
    print(base, results_dict[base])

with open("NEW_FLDATA_MULTI.csv", "w+") as f:
    f.write("Program,Probability,Bugs,Target Variables,Average FL Rank,Average ESP Rank, Score-FL, Score-ESP\n")
    for base in results_dict:
        fl_score = results_dict[base][0] / base[3]
        esp_score = results_dict[base][1] / base[3]
        f.write(base[0] + "," + str(base[1]) + "," + base[2] + "," + str(base[3]) +
                "," + str(results_dict[base][0]) + "," + str(results_dict[base][1]) + "," + str(fl_score) + "," + str(esp_score) + "\n")
