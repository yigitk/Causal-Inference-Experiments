import sys
import os

version_bug_dict = {
    "clausenLongFL": "r_0",
    "clausenLongFL.1": "d_1",
    "deriv_centralFL": "dy_0",
    "deriv_centralFL.1": "error_1",
    "frexpFL": "ex_0",
    "frexpFL.1": "f_0",
    "gsl_atanhFL": "to_return_0",
    "gsl_atanhFL.1": "to_return_2",
    "J0LongFL": "seps_0",
    "J0LongFL.1": "y_0",
    "J0LongFL.2": "ampl_0",
    "J0LongFL.3": "d_4",
    "J0LongFL.4": "ceps_0",
    "lngammaLongFL": "Ag_1",
    "lngammaLongFL.1": "x_9",
    "lngammaLongFL.2": "result_val_41",
    "lngammaLongFL.3": "Ag_0",
    "lngammaLongFL.4": "term2_0",
    "matric_transposeLongFL": "m_data_e2_1",
    "matric_transposeLongFL.1": "tmp_0",
    "skewnessFL": "mean_1",
    "skewnessFL.1": "variance_1",
    "synchrotron_1LongFL": "t_1",
    "synchrotron_1LongFL.1": "px_0",
    "synchrotron_1LongFL.2": "d_4",
    "synchrotron_1LongFL.3": "x_4",
    "synchrotron_1LongFL.4": "d_1",
    "transport_2LongFL": "result_val_5",
    "transport_2LongFL.1": "t_2"
}

version_target_vars_dict = {
    "clausenLongFL": 40,
    "clausenLongFL.1": 39,
    "deriv_centralFL": 28,
    "deriv_centralFL.1": 28,
    "frexpFL": 9,
    "frexpFL.1": 9,
    "gsl_atanhFL": 7,
    "gsl_atanhFL.1": 7,
    "J0LongFL": 40,
    "J0LongFL.1": 40,
    "J0LongFL.2": 40,
    "J0LongFL.3": 40,
    "J0LongFL.4": 40,
    "lngammaLongFL": 74,
    "lngammaLongFL.1": 74,
    "lngammaLongFL.2": 74,
    "lngammaLongFL.3": 74,
    "lngammaLongFL.4": 74,
    "matric_transposeLong": 24,
    "matric_transposeLong.1": 24,
    "skewness": 13,
    "skewness.1": 13,
    "synchrotron_1LongFL": 37,
    "synchrotron_1LongFL.1": 37,
    "synchrotron_1LongFL.2": 37,
    "synchrotron_1LongFL.3": 37,
    "synchrotron_1LongFL.4": 37,
    "transport_2Long": 47,
    "transport_2Long.1": 47
}

results_dict = {}

directory = os.fsencode(".")

for program in os.listdir(directory):
    if not os.path.isdir(program):
        continue
    for version in os.listdir(os.path.join(directory, program)):
        if "2Bug" in str(version) or "3Bug" in str(version):
            continue
        bug = version_bug_dict[str(version)[2:-1]]
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
                        bug_score = None
                        for line in f:
                            line_number += 1
                            rank = line_number - 4
                            variable = line.split(",")[1]
                            score = line.split(",")[2]
                            if variable == bug:
                                bug_score =FLoat(score)
                            all_file_ranks.append((variable,FLoat(score)))
                        ties = []
                        for i, tup in enumerate(all_file_ranks):
                            if tup[1] == bug_score:
                                ties.append(i)
                        rank = sum(ties) / len(ties) + 1
                        ranks.append(rank)
                average_rank = sum(ranks) / len(ranks)
                version_number = 1
                method_str = str(method)[2:-1]
                program_name = str(version)[2:-1]
                perc_str = str(perc)[2:-1]
                if ".1" in str(version):
                    version_number = 2
                    program_name = str(version)[2:-3]
                if ".2" in str(version):
                    version_number = 3
                    program_name = str(version)[2:-3]
                if ".3" in str(version):
                    version_number = 4
                    program_name = str(version)[2:-3]
                if ".4" in str(version):
                    version_number = 5
                    program_name = str(version)[2:-3]
                base = (program_name, version_number,
                        perc_str, bug, target_vars)
                if base not in results_dict:
                    results_dict[base] = [0, 0]
                    if method_str == FL":
                        results_dict[base][0] = average_rank
                    if method_str == "ESP":
                        results_dict[base][1] = average_rank
                else:
                    if method_str == FL":
                        results_dict[base][0] = average_rank
                    if method_str == "ESP":
                        results_dict[base][1] = average_rank

with open("NEWFLDATA.csv", "w+") as f:
    f.write("Program,Version,Probability,Bug,Target Variables,AverageFL Rank,Average ESP Rank, ScoreFL, Score-ESP\n")
    for base in results_dict:
       FL_score = results_dict[base][0] /FLoat(base[4])
        esp_score = results_dict[base][1] /FLoat(base[4])
        f.write(base[0] + "," + str(base[1]) + "," + base[2] + "," + base[3] + "," + str(
            base[4]) + "," + str(results_dict[base][0]) + "," + str(results_dict[base][1]) + "," + strFL_score) + "," + str(esp_score) + "\n")
