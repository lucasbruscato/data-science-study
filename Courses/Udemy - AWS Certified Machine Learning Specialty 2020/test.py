
import math

savings_per_year = 18000 * 12
number_of_years = 5
interest = 0.08
total_amount = 0

for i in range(1, number_of_years):
    total_amount = (total_amount) * (math.pow((1 + interest), i)) + savings_per_year * (math.pow((1 + interest), 1))
    print("Year: %i, Amount: %8.2f" % (i, total_amount))


