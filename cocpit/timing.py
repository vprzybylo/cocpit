import csv
import time


class Time:
    def __init__(self, epoch, since_total, since_epoch):
        self.epoch = epoch
        self.since_total = since_total
        self.since_epoch = since_epoch

    def write_times(self, model):
        """write time to train with respective modelname, epoch, and kfold to file"""
        time_elapsed = time.time() - self.since_total
        with open(
            "/data/data/saved_timings/model_timing_only_cpu.csv", "a", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow([model_name, self.epoch, kfold, time_elapsed])

    def print_time_all_epochs(self):
        """print time it took for all epochs to train"""
        time_elapsed = time.time() - self.since_total
        print(
            "All epochs comlete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

    def print_time_one_epoch(self, since_epoch):
        """print time it took for one epoch to complete"""
        time_elapsed = time.time() - since_epoch
        print(
            "Epoch complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
