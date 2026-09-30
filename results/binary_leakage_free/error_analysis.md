# Error analysis: Proposed hybrid (RF+XGB), leakage-free binary test set

## False positives

- 1,270 false positives out of 18,739 benign test flows (FPR 6.78%).
- 643 of them (50.6%) are benign-labelled flows to or from an external attacker host (184.98.36.245, 206.207.50.50). This may indicate label noise, but it is not proof: the attacker hosts also appear in 71 test flows of the attack-free Monday captures.
- 536 of them (42.2%) fall in the Monday captures, where no flow is labelled as an attack. These are genuine false alarms on normal traffic.
- FPR on benign flows that do not involve an attacker host: 3.80%.

### By capture file

|                               |   benign_flows |   false_positives | FPR     |
|:------------------------------|---------------:|------------------:|:--------|
| ('monday', False)             |           2586 |               238 | 9.20%   |
| ('monday', True)              |             71 |                23 | 32.39%  |
| ('monday-pvt', False)         |            926 |               275 | 29.70%  |
| ('public-thursday', False)    |           2768 |                44 | 1.59%   |
| ('public-thursday', True)     |             58 |                 4 | 6.90%   |
| ('public-tuesday', False)     |           3714 |                12 | 0.32%   |
| ('public-tuesday', True)      |           1497 |               582 | 38.88%  |
| ('public-wednesday', False)   |           2495 |                20 | 0.80%   |
| ('public-wednesday', True)    |            115 |                21 | 18.26%  |
| ('pvt-thursday', False)       |            513 |                 0 | 0.00%   |
| ('pvt-tuesday', False)        |            682 |                 3 | 0.44%   |
| ('pvt-wednesday', False)      |            402 |                 3 | 0.75%   |
| ('pvt-wednesday', True)       |              1 |                 1 | 100.00% |
| ('tcpdump-friday', False)     |           1671 |                30 | 1.80%   |
| ('tcpdump-friday', True)      |            442 |                12 | 2.71%   |
| ('tcpdump-pvt-friday', False) |            725 |                 2 | 0.28%   |
| ('tcpdump-pvt-friday', True)  |             73 |                 0 | 0.00%   |

## Missed APT flows (false negatives)

- 127 APT test flows are missed. By count, most are Account Discovery (57 of 127).
- The rare, low-volume activities have the highest miss *rates* but contribute few misses in absolute terms.

| Activity               |   apt_flows |   missed | share of all misses   | miss rate   |
|:-----------------------|------------:|---------:|:----------------------|:------------|
| Account Discovery      |         595 |       57 | 44.9%                 | 9.6%        |
| Network Scan           |        2121 |       30 | 23.6%                 | 1.4%        |
| SQL Injection          |          22 |       11 | 8.7%                  | 50.0%       |
| Directory Bruteforce   |        2247 |       10 | 7.9%                  | 0.4%        |
| Account Bruteforce     |          40 |        7 | 5.5%                  | 17.5%       |
| Web Vulnerability Scan |         653 |        5 | 3.9%                  | 0.8%        |
| CSRF                   |           3 |        3 | 2.4%                  | 100.0%      |
| Command Injection      |           3 |        2 | 1.6%                  | 66.7%       |
| Backdoor               |           3 |        1 | 0.8%                  | 33.3%       |
| Privilege Escalation   |           4 |        1 | 0.8%                  | 25.0%       |
| Data Exfiltration      |           3 |        0 | 0.0%                  | 0.0%        |
