# Quarterly forecast — methodology limitation

This project trains and evaluates a single **one-step-ahead monthly model**: at any forecast
origin, it predicts only the single month that immediately follows. Every quarterly figure in
`quarterly_forecast.csv` is a **sum of three such one-step-ahead forecasts**, each produced at its
own origin (the month before the one it predicts) by the rolling back-test — never a separate
quarterly model, and never a single forecast covering all three months at once.

As a direct consequence, this MVP
**cannot forecast all three months of a quarter at the start of the quarter**.
The second and third months of any quarter can only be forecast once the preceding month's data
becomes available, one month at a time. A genuine start-of-quarter, three-month-ahead forecast
would require a separate multi-horizon or recursive forecasting approach; that is out of scope for
this MVP and is a candidate v2 extension (PRD §32, §50).

The rolling estimate for the current partial quarter is not an exception to this: it combines the
already-observed actual sales of the quarter's completed months with the single genuine
one-step-ahead forecast for the next month, and is therefore never treated as a scored, complete
quarter (`complete = False`, no `actual_sum`).
