# 首轮真实数据审计

使用 BTS Reporting Carrier On-Time Performance 的 2024 年 1 月完整月包试跑，再用 7 月完整月包复核。表中按 BTS 记录的计划出发机场归属，使用五问定稿的六类结局。两个月均在模型开发期。本报告列出数据审计和选点结果。 [BTS 月包目录](https://www.transtats.bts.gov/PREZIP/) · [统一研究口径](ANALYSIS_MODELING_REPORT.md)

## 1. 两个月的机场汇总

| 指标 | 2024 年 1 月 | 2024 年 7 月 |
| --- | ---: | ---: |
| 计划航段数 | 547,271 | 634,613 |
| 升格为身份待查的记录 | 0 | 0 |
| 完全重复记录 | 0 | 0 |
| 计划出发机场数 | 334 | 342 |
| 报告承运人 ID 数 | 15 | 15 |
| C 取消 | 20,389 | 18,372 |
| V 备降且未取消 | 1,512 | 2,303 |
| L 未备降抵达且长延误 | 46,133 | 75,651 |
| M 未备降抵达且其他延误 | 80,277 | 106,214 |
| T 未备降且准点抵达 | 398,960 | 432,073 |
| U 主结局未知 | 0 | 0 |
| 全部计划航段中未备降且准点抵达 | 72.90% | 68.08% |
| 未备降抵达航段中的准点比例 | 75.94% | 70.38% |
| 已确认曾发生备降的航段 | 1,518 | 2,303 |
| 已确认最终抵达的航段 | 526,638 | 616,128 |
| 已确认最终抵达且晚至少 60 分钟 | 47,389 | 77,800 |

两个月的计划航段分母合计 **1,181,884**。一月改用未备降抵达航段作分母时，准点率上升约 **3.04 个百分点**。七月相差约 **2.29 个百分点**。

| 严重中断阈值 | 2024 年 1 月 | 2024 年 7 月 |
| --- | ---: | ---: |
| 45 分钟 | 82,508 | 117,688 |
| 60 分钟 | 68,034 | 96,326 |
| 90 分钟 | 51,146 | 69,318 |

结果文件为 [1 月汇总](../data/audit/2024-01/airport_summary.csv) 和 [7 月汇总](../data/audit/2024-07/airport_summary.csv)。其中 `ALL` 行为月包全部计划航段，其他各行按出发机场汇总。

## 2. 字段、边界和跨年核查

- 两份官方 ZIP 通过 CRC 校验。重新以标准库 `csv` 流式读取原始成员并独立计算六类结局，结果与月度程序一致。每个机场六类计数均加总至其计划航段数，各机场合计等于整月。
- `Flights` 在两个月全部为 `1.00`。候选航段键在两个月均未发现重复。两个月均有 -5、0、14、15、44、45、59、60、89、90 分钟的实际记录，已核对三个长延误阈值。两份月包均未触发取消与到达记录冲突，也未发现已抵达原目的地但 `DivArrDelay` 缺失的备降航段。测试样例覆盖这些未出现的边界。
- `DivAirportLandings=9` 在公开 CSV 中确实出现。1 月 132 条、7 月 41 条，均带取消标记；其中 1 月有 6 条还记录在始发机场以外的机场落地，7 月未发现该组合。这 6 条主结局为 C，同时计入已确认曾发生备降。两个月的原始 `Cancelled=1` 和 `Diverted=1` 同时为真的计数均为零，过程审计仍捕捉到交叉记录。 [1 月异常明细](../data/audit/2024-01/exceptions.csv) · [7 月异常明细](../data/audit/2024-07/exceptions.csv)
- 2023、2025、2026 年 1 月从官方 ZIP 读取首个 CSV 的表头。加上已完整读取的 2024 年 1 月和 7 月，共五个包的表头一致，包括末尾空列。跨年探针只读取表头，年度承运人覆盖和整月字段类型随逐月扩库验收。 [跨年表头探针](../data/audit/cross_year_schema_probe.json)
- 本批规范化输出保留本地 `hhmm` 及 `2400` 的日期偏移。开发期 3 月和 11 月的夏令时实测见第 7 节。

## 3. 首批八个主分析机场

采用 **ATL、DEN、DFW、MCO、ORD、PDX、RDU、SFO**。FAA 2024 年分类中 PDX、RDU 是中型枢纽，另六个是大型枢纽。这组机场覆盖东南部、中南部、山地、五大湖及太平洋沿岸，报告航司集中度和航程构成也有差异。选择与验收使用机场规模、地域及计划航线结构。 [FAA 2024 年上机人次和枢纽分类](https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/arp-cy2024-commercial-service-enplanements.pdf)

试跑的结构门槛为每月不少于 1,000 个计划出发航段、20 个目的地，且至少 50% 出发航段的目的地也由另外两个候选机场服务。八个机场在两个月均通过。共同目的地航段比例在 1 月为 **79.50%–100.00%**，7 月为 **82.01%–100.00%**。逐对比较仍可能缺乏重叠。按两机场到共同目的地的航段占比取较小值，最低一对在 1 月为 **39.76%**，7 月为 **40.61%**。第 2 问应继续报告逐对可比范围。正式扩库时复核该固定样本的跨月覆盖。 [八机场结构与覆盖表](../data/audit/airport_candidate_coverage.csv)

## 4. 气象档案小样

- **TAF。** 成功取到 ORD 和 ATL 在 2024 年 1 月 8 日 UTC 发布的结构化档案，两站各有 4 条定时预报。含修订和预报分段时，ORD 为 58 行、ATL 为 84 行。预报样本保留了 IEM 的发布时间、分段时间与报文 ID。 [ORD TAF](../data/raw/iem/taf_ORD_2024-01-08.csv) · [ATL TAF](../data/raw/iem/taf_ATL_2024-01-08.csv) · [TAF 取数记录](../data/raw/iem/taf_pilot_provenance.json)
- **GFS-MOS。** 同两站均成功取得 2024 年 1 月 8 日 12 UTC 模式周期，各 21 个预报时刻，有效时间从 1 月 8 日 18 UTC 到 1 月 11 日 12 UTC。档案列提供 `runtime` 和 `ftime`。核实实际发布时间后才能做严格的信息时点匹配。 [MOS 取数记录](../data/raw/iem/mos_pilot_provenance.json) · [IEM MOS 文档](https://mesonet.agron.iastate.edu/mos/)
- 此次确认两站可取到样本。TAF 完整原报文的取数接口遇到证书链错误，航段级提前 24 小时双端覆盖率尚未核实。公开计划信息基线可以先用航班数据构建。 [TAF 档案说明](https://mesonet.agron.iastate.edu/request/taf.php)

## 5. 文件与复现

- [单月审计程序](../code/bts_monthly.py) 接受单月 CSV 或含一个 CSV 的 BTS ZIP，生成 `flights.csv.gz`、`airport_summary.csv`、`exceptions.csv` 和 `manifest.json`。标准化航段表保留原始判定字段、过程标记、质量标记和 `source_row`。
- [1 月复现清单](../data/audit/2024-01/manifest.json) 和 [7 月复现清单](../data/audit/2024-07/manifest.json) 记录来源 URL、下载日期、ZIP 与脚本的 SHA-256、字段字典、运行命令及环境。使用 VS Code 全局 Python **3.12.10** 和现有 pandas **3.0.3**，航段结果采用 gzip 压缩 CSV。
- [边界与流程测试](../tests/test_bts_monthly.py) 覆盖时间格式、三档阈值、部分可判定结局、备降与取消过程、完全重复记录和异常 `Flights`。以下命令在项目根目录运行，使用一个新的结果目录重跑。

```powershell
$python = 'C:\Users\zhaozhiyi\AppData\Local\Programs\Python\Python312\python.exe'
$monthPackage = 'On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_1.zip'
& $python -B -X utf8 code/bts_monthly.py "data/raw/bts/$monthPackage" --month 2024-01 --output 'data/audit/recheck-2024-01' --source-url "https://www.transtats.bts.gov/PREZIP/$monthPackage" --downloaded-on 2026-11-26
& $python -B -m pytest -q -p no:cacheprovider tests/test_bts_monthly.py
```

## 6. 批量扩库前复测

新增 [批处理入口](../code/bts_batch.py)，从已下载月包生成八机场分析表。表中保留计划出发地或计划目的地命中八场的航段，并分别标记两种样本身份。逐条对照原始 CSV 确认五类延误原因、计划和实际耗时、距离及星期字段一致。按源行核对的六类主结局和 45、60、90 分钟阈值也与首轮相同，原机场汇总逐表一致。

| 源文件月份 | 八机场出发 | 八机场计划到达 | 两端均命中 | 精简表总行数 |
| --- | ---: | ---: | ---: | ---: |
| 2024-01 | 126,761 | 126,779 | 16,530 | 237,010 |
| 2024-07 | 148,412 | 148,403 | 20,083 | 276,732 |
| 合计 | 275,173 | 275,182 | 36,613 | 513,742 |

复测月包没有跨月候选键冲突，八机场出发航段没有主结局未知记录。专门的测试样例覆盖同一航段出现在两个源月包的情形，并确认月外记录可通过候选键和源行追溯。同一个批次再次运行可复用来源和脚本散列均匹配的月度结果。 [月度覆盖](../data/analysis/pilot/monthly_coverage.csv) · [承运人覆盖](../data/analysis/pilot/carrier_coverage.csv) · [未知结局分组](../data/analysis/pilot/unknown_by_origin_carrier_month.csv) · [跨月待查明细](../data/analysis/pilot/cross_month_issues.csv) · [复测清单](../data/analysis/pilot/manifest.json)

```powershell
& $python -B -X utf8 code/bts_batch.py --raw-dir data/raw/bts --output data/analysis/pilot --months 2024-01 2024-07
& $python -B -m pytest -q -p no:cacheprovider tests
```

正式开发期批次使用 `--years 2023 2024` 和独立输出目录。程序在处理前校验缺包，也可用 `--months` 指定分批处理中明确的月份。

## 7. 2023–2024 年开发期扩库

2023 年 1 月至 2024 年 12 月的 24 个 BTS 月包已全部入库。逐月核对压缩包大小、完整性和 SHA-256，确认原始表头一致。2024 年 1 月和 7 月的精简航段文件与第 6 节的试跑文件逐字节一致。 [BTS 月包目录](https://www.transtats.bts.gov/PREZIP/) · [下载脚本](../code/bts_fetch_development.py) · [批次清单](../data/analysis/development/manifest.json) · [月度覆盖](../data/analysis/development/monthly_coverage.csv)

| 年份 | BTS 有效航段 | 八机场出发 | 八机场计划到达 | 进出港合并后保留 |
| --- | ---: | ---: | ---: | ---: |
| 2023 | 6,847,899 | 1,587,731 | 1,587,671 | 2,960,527 |
| 2024 | 7,079,060 | 1,659,620 | 1,659,572 | 3,099,029 |
| 合计 | 13,926,959 | 3,247,351 | 3,247,243 | 6,059,556 |

24 个月的报告承运人 ID 集合相同，均为 15 家，与两年 BTS 指令列示的数量一致。逐月核对了六类结局加总、出发分母及进出港合并计数。跨月候选键核查没有发现待查记录。 [2023 年 BTS 指令](https://www.bts.gov/explore-topics-and-geography/modes/aviation/number-37-technical-reporting-directive-reporting-air) · [2024 年 BTS 指令](https://www.bts.gov/explore-topics-and-geography/modes/aviation/number-38-technical-reporting-directive-reporting-air) · [承运人覆盖](../data/analysis/development/carrier_coverage.csv) · [跨月待查明细](../data/analysis/development/cross_month_issues.csv)

八机场出发总体有 1 条未知结局记录，2023 年 10 月 ORD→DCA 的源行 400446 中，返航代码 9 与取消标记为 0 及有到达记录冲突，按既定规则归 $U$ 并计入出发分母。另有 1 条 2024 年 8 月 MIA→ATL 原始记录缺少航班号，无法确认航段身份，已留在异常表，未计入有效航段。 [未知结局分组](../data/analysis/development/unknown_by_origin_carrier_month.csv) · [2023 年 10 月异常](../data/analysis/development/2023-10/exceptions.csv) · [2024 年 8 月异常](../data/analysis/development/2024-08/exceptions.csv)

夏令时审计在 3 月和 11 月切换日及其前后各一天检查八机场之间的 7,007 条航段。按计划离港当地时间生成 UTC 候选，再用计划轮挡耗时与目的地计划到达钟点核对。6,994 条可唯一匹配，其中 1 条重复的秋季离港时刻由到达钟点消歧。全部匹配项的截止时点均为离港 UTC 前 24 个实际小时。其余 13 条均在春季切换日前一天，推算与报送到达钟点有差异，其中 12 条相差 60 分钟，1 条相差 1 分钟。异常保留源行和标记，第 1 问继续使用 BTS 报送结局及延误分钟。 [审核汇总](../data/analysis/development/dst_clock_summary.csv) · [逐航段明细](../data/analysis/development/dst_clock_checks.csv.gz) · [检查脚本](../code/bts_dst_audit.py) · [BTS 第 31 号指令](https://www.bts.gov/topics/airlines-and-airports/number-31-technical-directive-time-reporting-effective-jan-1-2019) · [NIST 夏令时规则](https://www.nist.gov/pml/time-and-frequency-division/popular-links/daylight-saving-time-dst)

开发期数据准备已验收。按月压缩 CSV、分组覆盖表和异常明细已用于第 1 问计算。 [问题一开发期统计](../outputs/问题一计算结果/airport_metrics.csv)

## 8. 问题二 · 排名与航班构成

2023–2025 年八场共有 4,943,437 条计划出发航段。48 格直接标准化后，PDX 的 $R_1$ 与 $D_{60}$ 名次集合均为第 1。两项指标各有 3 对点估计换位，未发现由同时区间证实的方向换位。调整后区间含零的机场对为 5/28 和 8/28。五家公共承运人的联合支撑内，ATL 和 DFW 的 $R_1$ 相对同样本粗率分别变化 −1.15 和 +4.79 个百分点，各覆盖 43.93% 和 53.36%。SFO–ATL 的严重中断率冬夏差距变化在固定时段与航距后为 −6.28 个百分点，95% 同时区间 [−11.12, −1.44]。2025 年全量两项粗率均未重现冬夏方向换位。8 场、288 个机场月与问题一对账一致，两类主推断各有 1,000 次有效重抽。问题二专用测试 3 个用例通过。 [问题二结果与结论](问题二计算结果.md) · [三年运行清单](../outputs/问题二计算结果/history_2023_2025/manifest.json) · [调整前后差异图](../figures/问题二计算结果/history_2023_2025/rank_adjustment_intervals.pdf) · [冬夏变化图](../figures/问题二计算结果/history_2023_2025/seasonal_variation_and_reversal.pdf)

## 9. 问题三 · 2025 年留出扩库与冻结评估

问题三冻结在 2024 年 9 月及此前月包。2025 年 12 个月包通过大小与 ZIP 校验，标准化批次覆盖八场计划出发 1,696,086 条、计划到达 1,696,027 条，跨月候选键核查无异常。出发有 5 条未知结局，五态概率通过 1,696,086 条逐行合法性和预测时点校验。冻结留出中，Info-A 的严重中断 AP 为 0.1904，Brier 为 0.093714。八场预报气温增强版的 Brier 为 0.093098，同航段配对差为 −0.000616，七日块 95% 区间 [−0.000718, −0.000171]。2025 年冻结预测移交问题四。 [2025 批次清单](../data/analysis/q3_holdout2025/manifest.json) · [问题三结果](问题三计算结果.md) · [冻结模型清单](../outputs/问题三计算结果/frozen_manifest.json) · [留出评估清单](../outputs/问题三计算结果/2025/evaluation_manifest.json)

问题三冻结后，以同一出发口径生成问题一、问题二的 2023–2025 历史版。4,943,437 个计划出发代表 288 个机场月。问题一独立复算 36 个月包中的 438 条原始记录。问题二同版本的计划到达为 4,943,270 条。 [问题一三年结果](问题一计算结果.md) · [问题一运行清单](../outputs/问题一计算结果/history_2023_2025/manifest.json) · [问题二回填清单](../outputs/问题二计算结果/history_2023_2025/manifest.json)

2026 年 1–9 月 9 个已发布月包完成独立后测，八场计划出发为 1,313,115 条，已判明 1,313,102 条，实际严重中断率 11.93%。同期复核中，2025、2026 年 1–9 月的 Info-A 校准截距分别为 −0.113、−0.357，斜率为 1.056、0.899。所有预测通过逐行验收，续用同一冻结模型散列。Info-A 的严重中断 Brier 为 0.102235，Info-B 为 0.104140，两者全量配对差为 +0.001905，95% 区间 [+0.000391, +0.002978]。 [2026 原始批次清单](../data/analysis/q3_later2026/manifest.json) · [2026 评估清单](../outputs/问题三计算结果/2026/evaluation_manifest.json) · [完整计算结果](问题三计算结果.md)

两年 1–9 月 Info-A 的 Brier 增加 0.007264，其中实测率不确定性项增加 0.006646，占观察增量的 91.5%。在两年每月均报送的 12 家承运人内，校准斜率仍由 1.063 降至 0.896。该事后复核使用各期实际班量。 [同期分解与承运人复核](../outputs/问题三计算结果/same_period_brier_and_carriers.csv)
