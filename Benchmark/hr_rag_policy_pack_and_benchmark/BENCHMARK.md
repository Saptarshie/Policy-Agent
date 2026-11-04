# Multihop RAG Benchmark: 20 Queries

These questions are grounded only in the four fictional Northstar Dynamics policy documents. Exact wording may vary, but answers must preserve the stated decision, amount, conditions, and citations. MH-19 and MH-20 must be refused.

## MH-01
**Question:** A Hybrid employee buys a ₹14,000 ergonomic chair and a ₹17,000 monitor 20 days after joining. How much can they claim immediately?

**Ground truth:** ₹0 immediately. Although the items are individually within their caps and their combined reimbursement would later be capped at ₹25,000, the employee becomes eligible only after completing 30 calendar days of employment.

**Expected behavior:** `answer`

**Ground-truth citations:**
- `03_WORKPLACE_EXPENSE_AND_REMOTE_WORK_POLICY.md` → `3.1 Eligibility and amount`
- `03_WORKPLACE_EXPENSE_AND_REMOTE_WORK_POLICY.md` → `3.2 Eligible items`

**Why multihop/complex:** Combine waiting-period eligibility, item caps, and overall cap.

## MH-02
**Question:** An Enhanced-tier employee has a pre-authorized dental implant costing ₹70,000 and clinically required bone grafting costing ₹10,000. What is the maximum plan payment?

**Ground truth:** ₹25,000. Enhanced pays 50% of eligible cost but is capped at ₹25,000 per implant, and clinically required bone grafting counts toward the same per-implant cap.

**Expected behavior:** `answer`

**Ground-truth citations:**
- `02_HEALTH_AND_WELLBEING_BENEFITS.md` → `4.2 Dental implants`

**Why multihop/complex:** Apply percentage, per-implant cap, and graft aggregation.

## MH-03
**Question:** A Remote employee is on paid caregiver leave for all of August and does no work. Can they claim August broadband, and does employer health coverage continue?

**Ground truth:** They cannot claim August broadband because they performed no working day in that month. Employer health coverage continues during paid caregiver leave.

**Expected behavior:** `answer`

**Ground-truth citations:**
- `03_WORKPLACE_EXPENSE_AND_REMOTE_WORK_POLICY.md` → `4. Monthly remote-work support`
- `01_LEAVE_AND_ATTENDANCE_POLICY.md` → `7.3 Benefit interaction`

**Why multihop/complex:** Join leave status with separate rules for attendance-based remote support and health continuation.

## MH-04
**Question:** An employee assigned to Guwahati works from Kolkata during a Guwahati holiday while on a temporary assignment whose letter does not change holiday treatment. Which holiday calendar applies?

**Ground truth:** The Guwahati holiday calendar applies. The HR-assigned location controls unless the temporary-assignment letter explicitly says the temporary location's holidays apply.

**Expected behavior:** `answer`

**Ground-truth citations:**
- `01_LEAVE_AND_ATTENDANCE_POLICY.md` → `9. Holidays and weekends during leave`
- `04_TRAVEL_RELOCATION_AND_MOBILITY_POLICY.md` → `12. Temporary assignments`

**Why multihop/complex:** Resolve the default rule and temporary-assignment exception.

## MH-05
**Question:** A Standard-tier employee asks whether a dental implant needed after an accident is covered. What should the assistant answer?

**Ground truth:** It is not covered under the Standard tier. The policy lists Standard dental implants as not covered and provides no accident exception for implant coverage.

**Expected behavior:** `answer`

**Ground-truth citations:**
- `02_HEALTH_AND_WELLBEING_BENEFITS.md` → `4.2 Dental implants`

**Why multihop/complex:** Use explicit table cell and avoid inventing an exception.

## MH-06
**Question:** A Remote employee submits a ₹1,400 broadband bill 40 days after payment. What amount is potentially reimbursable and whose extra approval is needed?

**Ground truth:** Up to ₹1,400 is potentially reimbursable because the Remote monthly cap is ₹1,500. Because submission is 31–60 days late, department-head approval is additionally required.

**Expected behavior:** `answer`

**Ground-truth citations:**
- `03_WORKPLACE_EXPENSE_AND_REMOTE_WORK_POLICY.md` → `4. Monthly remote-work support`
- `03_WORKPLACE_EXPENSE_AND_REMOTE_WORK_POLICY.md` → `2. Standard submission rules`

**Why multihop/complex:** Combine category cap with late-claim approval window.

## MH-07
**Question:** An employee with dependants received ₹1,20,000 in relocation benefits and voluntarily resigns after 8 completed months. How much must be repaid?

**Ground truth:** ₹60,000. Voluntary resignation after 6 but before 12 completed months requires 50% repayment, and 50% of ₹1,20,000 is ₹60,000.

**Expected behavior:** `answer`

**Ground-truth citations:**
- `04_TRAVEL_RELOCATION_AND_MOBILITY_POLICY.md` → `11. Relocation repayment`

**Why multihop/complex:** Map tenure to repayment band and calculate the amount.

## MH-08
**Question:** An employee ends the year with 8 unused Casual Leave days and 35 unused Earned Leave days. How many of each can carry into the next year?

**Ground truth:** 5 Casual Leave days and 30 Earned Leave days carry forward. The remaining 3 Casual Leave days expire; treatment of Earned Leave balances above the overall 48-day cap depends on the resulting balance and March encashment rule.

**Expected behavior:** `answer`

**Ground-truth citations:**
- `01_LEAVE_AND_ATTENDANCE_POLICY.md` → `4.2 Carry-forward`
- `01_LEAVE_AND_ATTENDANCE_POLICY.md` → `5.2 Carry-forward and encashment`

**Why multihop/complex:** Retrieve and distinguish two leave-category rules.

## MH-09
**Question:** A Premier member uses 3 EAP counseling sessions and 18 insured mental-health sessions in the same policy year. Can they receive one more insured session?

**Ground truth:** Yes. Premier allows 20 insured mental-health sessions per covered person per policy year, and EAP sessions do not consume the insurance session limit. After 18 insured sessions, 2 insured sessions remain.

**Expected behavior:** `answer`

**Ground-truth citations:**
- `02_HEALTH_AND_WELLBEING_BENEFITS.md` → `2. Annual coverage limits`
- `02_HEALTH_AND_WELLBEING_BENEFITS.md` → `5. Mental health and neurodevelopmental care`

**Why multihop/complex:** Separate EAP sessions from insurance limits and compute remainder.

## MH-10
**Question:** A Hybrid employee on an approved 40-day temporary assignment has hotel internet included in a reimbursed lodging bill. Can they also claim the same internet under monthly remote-work support?

**Ground truth:** No. Internet provided by temporary lodging is a travel expense, and the same cost cannot also be claimed as remote-work internet support or reimbursed twice.

**Expected behavior:** `answer`

**Ground-truth citations:**
- `04_TRAVEL_RELOCATION_AND_MOBILITY_POLICY.md` → `12. Temporary assignments`
- `03_WORKPLACE_EXPENSE_AND_REMOTE_WORK_POLICY.md` → `8. Interaction with wellbeing benefits`

**Why multihop/complex:** Apply the travel-specific rule and general no-double-reimbursement principle.

## MH-11
**Question:** A caregiver takes 20 days of secondary caregiver leave in three blocks of 5, 5, and 10 working days. Is that allowed?

**Ground truth:** No. Although each block is at least 5 working days and totals 20 days, secondary caregiver leave may be taken in no more than two blocks.

**Expected behavior:** `answer`

**Ground-truth citations:**
- `01_LEAVE_AND_ATTENDANCE_POLICY.md` → `7.2 Secondary caregiver leave`

**Why multihop/complex:** Check total, minimum block length, and maximum number of blocks.

## MH-12
**Question:** A business-only flight costs ₹28,000, but adding two personal days makes the ticket ₹25,000. How much does the company pay, and who pays personal lodging?

**Ground truth:** The company pays ₹25,000, the lower of the business-only quote and actual combined itinerary. The employee pays all personal lodging and meals.

**Expected behavior:** `answer`

**Ground-truth citations:**
- `04_TRAVEL_RELOCATION_AND_MOBILITY_POLICY.md` → `7. Combining business and personal travel`

**Why multihop/complex:** Apply lower-of fare rule and separate personal costs.

## MH-13
**Question:** An employee wants a ₹10,000 treadmill desk, claiming ergonomics. Can it be reimbursed from either the home-office allowance or wellbeing wallet?

**Ground truth:** No. A treadmill desk is excluded from the home-office allowance as exercise equipment, and personal home-gym equipment is also excluded from the wellbeing wallet.

**Expected behavior:** `answer`

**Ground-truth citations:**
- `03_WORKPLACE_EXPENSE_AND_REMOTE_WORK_POLICY.md` → `3.3 Ineligible items`
- `02_HEALTH_AND_WELLBEING_BENEFITS.md` → `9.2 Ineligible wellbeing expenses`

**Why multihop/complex:** Cross-check two programmes to prevent category switching.

## MH-14
**Question:** An employee takes Sick Leave Monday through Wednesday and then requests short-term remote work for six working days because commuting remains difficult. What documentation may be required?

**Ground truth:** A medical certificate is required for the three consecutive Sick Leave days because Sick Leave exceeds two consecutive working days. A medical note may also be requested for the remote-work arrangement because it lasts longer than five working days.

**Expected behavior:** `answer`

**Ground-truth citations:**
- `01_LEAVE_AND_ATTENDANCE_POLICY.md` → `6.1 Sick Leave`
- `01_LEAVE_AND_ATTENDANCE_POLICY.md` → `10. Remote work during recovery`

**Why multihop/complex:** Combine two separate documentation thresholds.

## MH-15
**Question:** A 7-hour flight itinerary has two nonstop segments of 3.5 hours each. Is extra-legroom automatically allowed under the 6–10 hour rule?

**Ground truth:** No. Duration is assessed by the scheduled nonstop time of each booked segment, not total itinerary time. Each 3.5-hour segment falls under the under-6-hour Economy entitlement.

**Expected behavior:** `answer`

**Ground-truth citations:**
- `04_TRAVEL_RELOCATION_AND_MOBILITY_POLICY.md` → `2. Booking class`

**Why multihop/complex:** Interpret segment-level duration rather than total trip time.

## MH-16
**Question:** A Premier member's dependent child aged 17 needs medically necessary orthodontics costing ₹90,000 and has never used the benefit. What is the maximum stated coverage?

**Ground truth:** ₹75,000, subject to plan conditions. Premier covers medically necessary orthodontics for dependent children under 18 up to a lifetime maximum of ₹75,000 per child.

**Expected behavior:** `answer`

**Ground-truth citations:**
- `02_HEALTH_AND_WELLBEING_BENEFITS.md` → `4.3 Orthodontics`

**Why multihop/complex:** Check tier, age, medical necessity, and lifetime cap.

## MH-17
**Question:** A permanent employee spends ₹24,000 on one approved certification and resigns voluntarily 60 days after reimbursement. How much must they repay?

**Ground truth:** ₹12,000. A single learning expense above ₹20,000 triggers 50% repayment when the employee resigns within 90 calendar days; 50% of ₹24,000 is ₹12,000.

**Expected behavior:** `answer`

**Ground-truth citations:**
- `03_WORKPLACE_EXPENSE_AND_REMOTE_WORK_POLICY.md` → `6. Learning and certification budget`

**Why multihop/complex:** Apply threshold, resignation window, and repayment percentage.

## MH-18
**Question:** Can an employee expense a personal home gym if a doctor says exercise would improve general health?

**Ground truth:** No. Personal home-gym equipment is expressly excluded from both the wellbeing wallet and home-office allowance. A general medical recommendation is not the same as an approved accessibility accommodation.

**Expected behavior:** `answer`

**Ground-truth citations:**
- `02_HEALTH_AND_WELLBEING_BENEFITS.md` → `9.2 Ineligible wellbeing expenses`
- `03_WORKPLACE_EXPENSE_AND_REMOTE_WORK_POLICY.md` → `3.3 Ineligible items`
- `03_WORKPLACE_EXPENSE_AND_REMOTE_WORK_POLICY.md` → `7. Accessibility accommodations`

**Why multihop/complex:** Reject category laundering while recognizing the formal accommodation path.

## MH-19
**Question:** How many paid days off does the policy provide for an employee's wedding?

**Ground truth:** I don't have enough information in the uploaded policies to answer that. Please contact People Operations.

**Expected behavior:** `refuse`

**Ground-truth citations:**
- None. The correct behavior is refusal.

**Why multihop/complex:** Refusal: no wedding-leave entitlement is stated.

## MH-20
**Question:** What percentage of salary is contributed to the employee provident fund?

**Ground truth:** I don't have enough information in the uploaded policies to answer that. Please contact People Operations.

**Expected behavior:** `refuse`

**Ground-truth citations:**
- None. The correct behavior is refusal.

**Why multihop/complex:** Refusal: retirement or provident-fund contribution is not defined.
