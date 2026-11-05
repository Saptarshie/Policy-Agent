# Adversarial Policy-Arbitrage and Anti-Loophole Benchmark

**Status:** Fully synthetic and fictional  
**Documents:** 4 long Markdown instruments  
**Questions:** 20  
**Scope:** Tax-like rules, grants, employee benefits, ownership attribution, nominee arrangements, lawful elections, sham transactions, anti-fragmentation, and calibrated uncertainty.

> The benchmark is deliberately designed to distinguish lawful optimization expressly allowed by fictional rules from concealment, circular steps, sham ownership, double recovery, or abuse. It is not real-world tax or legal guidance.

## Scoring per question

- 3 points: correct bottom-line conclusion
- 3 points: all conditions, exceptions, and residual risks covered
- 2 points: correct cross-document reasoning where required
- 1 point: accurate citations
- 1 point: calibrated uncertainty and no real-world generalization

## AX-01

**Question:** Mira owns a consulting company with one major client generating 78% of revenue. She is the sole consultant, but the company owns expensive servers and may substitute qualified staff without client consent. Two unrelated engineers perform substantive delivery work. Is all company income automatically attributed to Mira under the personal-service rule?

**Ground truth:** No automatic attribution can be concluded. The three threshold conditions appear satisfied as to client concentration and Mira’s skill, and the company has fewer than three unrelated full-time substantive employees. However, the exception may apply because the company owns substantial non-personal assets, appears to bear commercial capacity, and can substitute qualified personnel without client approval. The facts must establish meaningful commercial risk as well. The correct answer is conditional, not a definite attribution or exemption.

**Reasoning modes:** multi-condition, exception, calibrated-uncertainty

**Ground-truth citations:**
- `01_FICTIONAL_REVENUE_CODE.md` → `3.2 Personal service entities`

## AX-02

**Question:** An employee asks a client to pay N$400,000 of earned bonus to a company owned by the employee’s adult sibling. Can the employee avoid current salary taxation because the money never entered the employee’s account?

**Ground truth:** No. Remuneration redirected at the employee’s request remains taxable to the employee when paid or made available. Routing it to a sibling’s company does not change the source or character. Depending on the sibling company’s rights and obligations, additional consequences may exist, but the employee cannot avoid the initial inclusion.

**Reasoning modes:** assignment-of-income, related-person

**Ground-truth citations:**
- `01_FICTIONAL_REVENUE_CODE.md` → `3.1 Employee remuneration`
- `01_FICTIONAL_REVENUE_CODE.md` → `1.3 Related persons`

## AX-03

**Question:** A business buys a N$280,000 integrated production system through three invoices of N$95,000, N$95,000, and N$90,000 and titles each component to a different controlled subsidiary. Can the group immediately deduct all three under the small-asset allowance?

**Ground truth:** No. Functionally integrated components acquired under one plan are treated as one item, and invoice, purchase-order, or title splitting does not create separate items. The grant rules also aggregate common-control entities and connected projects, reinforcing that the fragmentation has no independent effect. The N$280,000 system exceeds the N$100,000 per-item immediate-deduction threshold.

**Reasoning modes:** cross-regime, anti-fragmentation, arithmetic

**Ground-truth citations:**
- `01_FICTIONAL_REVENUE_CODE.md` → `4.3 Small-asset allowance`
- `02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT.md` → `2. Common control`
- `02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT.md` → `10. Anti-fragmentation and principal-purpose test`

## AX-04

**Question:** An FIC lends N$800,000 to its shareholder on 20 December. Before payment there is a written three-year market-rate agreement and adequate security, but the first principal installment is scheduled for February. Is dividend treatment avoided for that year?

**Ground truth:** No. All loan safe-harbor conditions must be satisfied, including at least one scheduled principal payment made by year-end. Because no principal payment was made by year-end, the loan is treated as a dividend at year-end.

**Reasoning modes:** conjunctive-safe-harbor, timing

**Ground-truth citations:**
- `01_FICTIONAL_REVENUE_CODE.md` → `5.3 Shareholder loans`

## AX-05

**Question:** On 28 December, an FIC shareholder repays a prior loan using money borrowed from a bank. On 15 January, the FIC issues a new substantially identical loan, and the shareholder uses it to repay the bank. Does the December repayment prevent deemed-dividend treatment?

**Ground truth:** No. The repayment is indirectly funded by a substantially identical new FIC loan within 90 days, so it is ignored. The temporary bank bridge does not break the linked refinancing chain.

**Reasoning modes:** step-transaction, 90-day-lookback, anti-avoidance

**Ground-truth citations:**
- `01_FICTIONAL_REVENUE_CODE.md` → `5.3 Shareholder loans`
- `01_FICTIONAL_REVENUE_CODE.md` → `9.2 Indicators`

## AX-06

**Question:** A taxpayer sells Fund A at a loss and buys Fund B the next day. Both are broad-market funds, highly correlated, but track different independently maintained indices. Is the loss automatically denied?

**Ground truth:** No. The Code expressly says broad-market funds tracking different independently maintained indices are not substantially identical merely because they are correlated. The loss is not automatically denied on the stated facts, though factual replication or another anti-avoidance arrangement could change the analysis.

**Reasoning modes:** lawful-optimization, safe-harbor, qualified-answer

**Ground-truth citations:**
- `01_FICTIONAL_REVENUE_CODE.md` → `6.3 Loss harvesting`
- `01_FICTIONAL_REVENUE_CODE.md` → `9.3 Safe tax planning`

## AX-07

**Question:** A taxpayer sells shares at a loss, and the taxpayer’s spouse buys the same shares two days later using an independently funded account. Can the household preserve the loss?

**Ground truth:** No under the specific loss-harvesting rule. A spouse is a related person, and the bed-and-spouse provision applies even where bank accounts are independent. The loss is denied and added to the replacement shares’ basis.

**Reasoning modes:** related-person, basis-carryover

**Ground-truth citations:**
- `01_FICTIONAL_REVENUE_CODE.md` → `6.3 Loss harvesting`
- `01_FICTIONAL_REVENUE_CODE.md` → `6.4 Bed-and-spouse rule`

## AX-08

**Question:** A donor gives N$1,000,000 to a PBO and receives a preferential supply contract worth N$100,000. The benefit is disclosed. What amount is potentially treated as a qualifying donation before applying the tax-liability cap?

**Ground truth:** N$900,000. The benefit exceeds the 2% disregard and therefore reduces the qualifying donation by its N$100,000 fair value. Because it was disclosed, the rule making the entire connected donation ineligible for deliberate concealment does not apply. The resulting credit is 25% of N$900,000, subject to the separate cap of 20% of pre-credit tax liability.

**Reasoning modes:** benefit-netting, calculation, cap-awareness

**Ground-truth citations:**
- `01_FICTIONAL_REVENUE_CODE.md` → `7. Donations and public-benefit credits`

## AX-09

**Question:** An investor subscribes N$1,000,000 for certified startup shares using a market-rate bank loan. The startup neither funds nor guarantees the loan. The investor receives ordinary dividends in year two. Is the investment credit recaptured?

**Ground truth:** No on the stated facts. A genuine market-rate bank loan not funded or guaranteed by the startup or promoter is permitted, and ordinary commercial dividends do not trigger recapture. The investor must still hold the shares for three years and avoid the other listed disqualifying protections or repayments.

**Reasoning modes:** lawful-chaining, safe-harbor, ongoing-conditions

**Ground-truth citations:**
- `01_FICTIONAL_REVENUE_CODE.md` → `8. Startup investment relief`

## AX-10

**Question:** A controlled group orders one green-equipment project in March but schedules half for delivery in March and half in April, crossing fiscal years, to claim two annual rebate caps. No technical or demand change occurred. Does the split multiply the cap?

**Ground truth:** No. A unified project cannot be fragmented across subsidiaries or fiscal years where a binding commitment or unified implementation plan existed before the split. Common-control caps apply to the enterprise group, and the principal-purpose anti-fragmentation rule independently supports aggregation.

**Reasoning modes:** fiscal-year-splitting, group-cap, substance-over-form

**Ground-truth citations:**
- `02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT.md` → `3.4 Connected acquisitions`
- `02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT.md` → `2. Common control`
- `02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT.md` → `10. Anti-fragmentation and principal-purpose test`

## AX-11

**Question:** After its first green-equipment installation has operated for eight months, a company faces genuinely unexpected demand and its board approves an unplanned second expansion. The company had no earlier commitment. Must both projects be aggregated?

**Ground truth:** Not necessarily. The Act expressly treats a genuine later expansion as separate where it responds to unanticipated demand, is board-approved after the first project became operational, and there was no earlier commitment. The common-control annual cap still applies if both fall in the same fiscal year.

**Reasoning modes:** legitimate-planning, exception, residual-cap

**Ground-truth citations:**
- `02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT.md` → `3.4 Connected acquisitions`
- `02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT.md` → `2. Common control`

## AX-12

**Question:** A group dismisses ten employees in the capital, then opens ten similarly functioning positions in a designated region two months later. Can it claim the Regional Employment Credit by using a new subsidiary?

**Ground truth:** No. Common-control entities are treated as one enterprise for headcount and caps. Roles transferred within the group are not net new, and replacing employees who left within six months is generally not net new unless the documented unrelated-elimination and materially-different-function exception applies. A new subsidiary does not reset the test.

**Reasoning modes:** entity-interposition, headcount-baseline, replacement-rule

**Ground-truth citations:**
- `02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT.md` → `2. Common control`
- `02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT.md` → `4.2 Qualifying job`
- `02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT.md` → `4.3 Transfers and replacements`

## AX-13

**Question:** A company receives a Green Equipment Rebate for a machine and also includes the machine’s full invoice as eligible research expenditure for the Innovation Grant because researchers use it 40% of the time. What is the maximum defensible treatment?

**Ground truth:** The same economic cost cannot be double funded. At most, a properly allocated, otherwise eligible research-use portion not already supported by the equipment rebate could be considered, and the rebate or other recovery must reduce the relevant supported cost. Whether any portion qualifies also depends on the equipment cost being eligible research expenditure rather than a capital purchase or pre-existing technology, which the facts do not establish.

**Reasoning modes:** double-funding, allocation, insufficient-facts

**Ground-truth citations:**
- `02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT.md` → `8. Stacking and double funding`
- `02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT.md` → `5.2 Eligible research`
- `02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT.md` → `5.4 Background intellectual property`

## AX-14

**Question:** Two spouses employed by Novacrest incur N$400,000 of eligible dependent-care expenses. Can each claim the N$180,000 household cap by paying separate invoices?

**Ground truth:** No. The household has one shared N$180,000 annual cap, and combined reimbursement also cannot exceed 50% of net cost. Fifty percent of N$400,000 is N$200,000, so the household cap controls at N$180,000. The spouses may allocate claims between them but cannot duplicate the cap.

**Reasoning modes:** household-aggregation, calculation, anti-duplication

**Ground-truth citations:**
- `03_EMPLOYEE_BENEFITS_AND_EQUITY_POLICY.md` → `5. Dependent-care benefit`
- `03_EMPLOYEE_BENEFITS_AND_EQUITY_POLICY.md` → `1. Eligibility and household aggregation`

## AX-15

**Question:** A Hybrid employee works remotely eight days and commutes on twelve days in a month. Public transport cost is N$5,500. Which benefit may the employee choose?

**Ground truth:** The employee is eligible for remote-work support of N$4,000 because the eight-day threshold is met, and for commuter support based on actual cost of N$5,500. The policy expressly allows choosing the more favorable option after month-end, but not both. The maximum choice is therefore N$5,500 of commuter support.

**Reasoning modes:** express-election, lawful-benefit-optimization, calculation

**Ground-truth citations:**
- `03_EMPLOYEE_BENEFITS_AND_EQUITY_POLICY.md` → `4. Commuter and remote-work support`
- `03_EMPLOYEE_BENEFITS_AND_EQUITY_POLICY.md` → `10. Anti-abuse rule`

## AX-16

**Question:** An employee enrolls in one N$120,000 certification but asks the provider for three N$40,000 invoices to avoid pre-approval and potential resignation repayment. Does that work?

**Ground truth:** No. One course cannot be divided into modules, invoices, or payment plans to avoid pre-approval. Multiple reimbursements arising from one enrollment agreement are aggregated, so the N$120,000 amount exceeds both the N$40,000 pre-approval threshold and the N$80,000 repayment threshold. If the employee resigns voluntarily within six months after reimbursement, 50% is repayable.

**Reasoning modes:** invoice-splitting, aggregation, repayment-trigger

**Ground-truth citations:**
- `03_EMPLOYEE_BENEFITS_AND_EQUITY_POLICY.md` → `2. Professional Development Account`

## AX-17

**Question:** An employee asks a colleague to submit the employee’s candidate referral in exchange for half the award. The colleague had never heard of the candidate before. Can the colleague keep the referral award?

**Ground truth:** No. The policy expressly prohibits routing a referral through a colleague in exchange for sharing the award. The colleague lacks the independent-knowledge exception, and the collusive arrangement is also recoverable under the anti-abuse rule.

**Reasoning modes:** collusion, sham-intermediary

**Ground-truth citations:**
- `03_EMPLOYEE_BENEFITS_AND_EQUITY_POLICY.md` → `6. Referral awards`
- `03_EMPLOYEE_BENEFITS_AND_EQUITY_POLICY.md` → `10. Anti-abuse rule`

## AX-18

**Question:** Before a change in control, an employee expects relocation of 100 km. The employee resigns immediately without giving Good Reason notice or allowing cure. Do unvested options accelerate?

**Ground truth:** No on the stated facts. Acceleration requires a qualifying termination or resignation for Good Reason within 12 months after the change in control. Good Reason also requires actual qualifying circumstances, written notice within 30 days, and a 30-day cure opportunity. An anticipatory resignation before the event and without process does not qualify.

**Reasoning modes:** sequence, condition-precedent, notice-and-cure

**Ground-truth citations:**
- `03_EMPLOYEE_BENEFITS_AND_EQUITY_POLICY.md` → `8.2 Change in control`

## AX-19

**Question:** A founder transfers 30% of a company outright to an adult sibling to bring the founder below a grant ownership threshold. The sibling pays fair value with independent financing, controls and votes the shares independently, bears all risk, and there is no return agreement. Must the transfer be ignored solely because benefit eligibility motivated it?

**Ground truth:** No. A genuine outright transfer is respected even if partly motivated by eligibility when the transferor retains no control, benefit, economic exposure, or reversal right. Tax or program motivation alone is insufficient. However, any program-specific common-control or related-person rule may still aggregate the parties for a separate purpose, so crossing the registry percentage threshold does not guarantee grant eligibility.

**Reasoning modes:** lawful-restructuring, substance, cross-statute-residual-risk

**Ground-truth citations:**
- `04_OWNERSHIP_TRUSTS_AND_DISCLOSURE_RULES.md` → `5. Temporary transfers`
- `04_OWNERSHIP_TRUSTS_AND_DISCLOSURE_RULES.md` → `9. Anti-circumvention`
- `04_OWNERSHIP_TRUSTS_AND_DISCLOSURE_RULES.md` → `8. Programme look-through`
- `02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT.md` → `2. Common control`

## AX-20

**Question:** A founder temporarily transfers shares to a nominee under an unwritten promise to return them after a grant is approved. The founder keeps dividends through a side payment and indemnifies the nominee against loss. Can the nominee be treated as the real owner for disclosure and grant thresholds?

**Ground truth:** No. The return understanding, retained economic benefit, and loss indemnity cause the temporary transfer to be ignored. Nominee holdings are attributed to the principal, and the side arrangement is a strong anti-circumvention indicator. The common-control and anti-fragmentation rules also allow the grant administrator to disregard the structure.

**Reasoning modes:** nominee, retained-benefit, anti-circumvention

**Ground-truth citations:**
- `04_OWNERSHIP_TRUSTS_AND_DISCLOSURE_RULES.md` → `2. Indirect ownership`
- `04_OWNERSHIP_TRUSTS_AND_DISCLOSURE_RULES.md` → `5. Temporary transfers`
- `04_OWNERSHIP_TRUSTS_AND_DISCLOSURE_RULES.md` → `9. Anti-circumvention`
- `02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT.md` → `10. Anti-fragmentation and principal-purpose test`
