from collections import Counter


class FailureAwareCurriculum:
    def __init__(self):
        self.failure_counts = Counter()

    def update(self, diagnoses):
        for item in diagnoses:
            diagnosis = item[1] if isinstance(item, tuple) else item
            self.failure_counts[diagnosis.failure_type] += 1

    def weights(self):
        total = sum(self.failure_counts.values())
        if total == 0:
            return {}
        return {key: value / total for key, value in self.failure_counts.items()}
