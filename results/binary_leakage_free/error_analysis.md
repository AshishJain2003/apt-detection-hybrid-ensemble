# Error analysis: Proposed hybrid (RF+XGB), leakage-free binary test set

- False positives: 1,122 of 18,737 benign test flows (FPR 5.99%).
- 616 of those false positives (54.9%) are 'benign'-labelled flows to/from an external attacker host (184.98.36.245, 206.207.50.50).
- FPR on benign flows that do not involve an attacker host: 3.07%.

## False positives by capture file

|                               |   benign_flows |   false_positives | FPR    |
|:------------------------------|---------------:|------------------:|:-------|
| ('monday', False)             |           2586 |               174 | 6.73%  |
| ('monday', True)              |             71 |                18 | 25.35% |
| ('monday-pvt', False)         |            927 |               235 | 25.35% |
| ('public-thursday', False)    |           2768 |                38 | 1.37%  |
| ('public-thursday', True)     |             58 |                 2 | 3.45%  |
| ('public-tuesday', False)     |           3714 |                 7 | 0.19%  |
| ('public-tuesday', True)      |           1497 |               577 | 38.54% |
| ('public-wednesday', False)   |           2501 |                19 | 0.76%  |
| ('public-wednesday', True)    |            115 |                13 | 11.30% |
| ('pvt-thursday', False)       |            521 |                 0 | 0.00%  |
| ('pvt-tuesday', False)        |            669 |                 5 | 0.75%  |
| ('pvt-wednesday', False)      |            403 |                 1 | 0.25%  |
| ('pvt-wednesday', True)       |              2 |                 1 | 50.00% |
| ('tcpdump-friday', False)     |           1680 |                22 | 1.31%  |
| ('tcpdump-friday', True)      |            427 |                 5 | 1.17%  |
| ('tcpdump-pvt-friday', False) |            725 |                 5 | 0.69%  |
| ('tcpdump-pvt-friday', True)  |             73 |                 0 | 0.00%  |

## Missed APT flows by activity

| Activity               |   apt_flows |   missed | miss rate   |
|:-----------------------|------------:|---------:|:------------|
| Account Bruteforce     |          47 |       10 | 21.3%       |
| Account Discovery      |         602 |       93 | 15.4%       |
| Backdoor               |           3 |        1 | 33.3%       |
| CSRF                   |           3 |        3 | 100.0%      |
| Command Injection      |           4 |        3 | 75.0%       |
| Data Exfiltration      |           2 |        0 | 0.0%        |
| Directory Bruteforce   |        2233 |       11 | 0.5%        |
| Network Scan           |        2123 |       44 | 2.1%        |
| Privilege Escalation   |           5 |        2 | 40.0%       |
| SQL Injection          |          19 |        7 | 36.8%       |
| Web Vulnerability Scan |         653 |        4 | 0.6%        |
