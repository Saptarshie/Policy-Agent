# Complex Contract Reasoning Benchmark

**Corpus:** Four interlocking fictional enterprise agreements  
**Questions:** 20  
**Purpose:** Evaluate hierarchy resolution, multihop retrieval, deadlines, calculations, exception handling, remedy chaining, formalities, anti-loophole reasoning, and calibrated uncertainty.

> This benchmark uses synthetic contracts and is not legal advice. Answers should be judged only against the supplied corpus.

## Suggested scoring

- **4 points:** Substantive conclusion and calculation are correct.
- **3 points:** Reasoning applies the correct hierarchy, exceptions, and conditions.
- **2 points:** Citations identify the correct document and section.
- **1 point:** The answer avoids overclaiming when facts are insufficient.

## LC-01
**Question:** In November 2026, Customer has 2,040 regular active users and 55 machine-only service accounts. How many accounts are billable as overage and what is the monthly charge?

**Ground truth:** CO-2 applies because it expressly replaced OF-7 Section 2.3. There are 40 regular excess users plus 5 service accounts above the 50-account exclusion, for 45 excess accounts. The first 10 are waived, so 35 are billable at ₹800 each, producing a ₹28,000 monthly overage.

**Reasoning tags:** hierarchy, arithmetic, exception

**Required citations:**
- `02_ENTERPRISE_ORDER_FORM_AND_CHANGE_ORDERS.md` → `CO-2.1 Revised User Overage`
- `01_MASTER_SERVICES_AGREEMENT.md` → `1.2 Order of Precedence`

## LC-02
**Question:** Customer sent a renewal non-renewal notice 70 days before 31 January 2029. Does OF-7 renew?

**Ground truth:** Yes. OF-7 requires at least 90 calendar days' notice for the first renewal after the Initial Term. The MSA's 60-day default does not control because the Order Form ranks above the MSA. A notice sent only 70 days before expiry is late, so OF-7 renews for the specified 12-month period.

**Reasoning tags:** hierarchy, deadline

**Required citations:**
- `02_ENTERPRISE_ORDER_FORM_AND_CHANGE_ORDERS.md` → `5.1 Renewal`
- `01_MASTER_SERVICES_AGREEMENT.md` → `1.2 Order of Precedence`
- `01_MASTER_SERVICES_AGREEMENT.md` → `4.2 Order Terms`

## LC-03
**Question:** Provider's annual fee notice arrives 68 days before renewal and proposes a permitted 4% increase. Can the increase apply at the imminent renewal?

**Ground truth:** No. Even if 4% is within the substantive cap, the MSA requires at least 75 days' notice. A late notice shifts the increase to the following renewal.

**Reasoning tags:** deadline, condition

**Required citations:**
- `01_MASTER_SERVICES_AGREEMENT.md` → `4.3 Renewal Pricing`

## LC-04
**Question:** Customer used the production system with 180 users for 20 consecutive business days in December 2026 and never issued acceptance. Is Production Acceptance deemed accepted through productive use?

**Ground truth:** No, not under the revised productive-use trigger. CO-2 requires at least 250 Named Users on each of 15 consecutive business days. Although the duration exceeds 15 days, 180 users is below the threshold. Deemed acceptance may still occur if Customer fails to reject within the separate 20-business-day Acceptance Period, but the facts do not establish when that period expired.

**Reasoning tags:** temporal, threshold, insufficient-facts

**Required citations:**
- `02_ENTERPRISE_ORDER_FORM_AND_CHANGE_ORDERS.md` → `CO-2.2 Revised Production Acceptance`

## LC-05
**Question:** Customer has paid the ₹18,00,000 allocated to the Rules Engine but is withholding an unrelated implementation invoice under a valid good-faith dispute. Who owns the Rules Engine?

**Ground truth:** Customer owns the Rules Engine. CO-2 makes ownership transfer upon payment of the specifically allocated ₹18,00,000 even if an unrelated implementation invoice is disputed in good faith. Provider still owns the excluded Background Materials embedded in or used with the custom work.

**Reasoning tags:** hierarchy, ownership, carve-out

**Required citations:**
- `02_ENTERPRISE_ORDER_FORM_AND_CHANGE_ORDERS.md` → `CO-2.3 Custom Work Payment`
- `02_ENTERPRISE_ORDER_FORM_AND_CHANGE_ORDERS.md` → `Appendix 1: Store Replenishment Rules Engine`
- `01_MASTER_SERVICES_AGREEMENT.md` → `7.4 Customer-Owned Custom Work`

## LC-06
**Question:** Availability is 97.8% in one month for the Inventory Optimization Module only. What contractual remedies arise if Customer acts on time?

**Ground truth:** The month earns a 20% availability credit calculated on one-twelfth of that module's annual recurring fee. It also constitutes Chronic Failure because availability fell below 98.00% in a single month. Customer may give legal notice within 30 days after the condition is met and terminate that module, with an effective date 15–60 days after notice, receiving a pro-rata refund of prepaid unused fees. It cannot terminate the unaffected base platform solely on these facts.

**Reasoning tags:** remedy-chaining, calculation-base, scope

**Required citations:**
- `03_SERVICE_LEVEL_AND_SUPPORT_SCHEDULE.md` → `4. Availability Credits`
- `03_SERVICE_LEVEL_AND_SUPPORT_SCHEDULE.md` → `8. Chronic Failure`
- `03_SERVICE_LEVEL_AND_SUPPORT_SCHEDULE.md` → `12. Service Measurement Across Modules`
- `02_ENTERPRISE_ORDER_FORM_AND_CHANGE_ORDERS.md` → `2.1 Recurring Fees`

## LC-07
**Question:** The base platform and both modules were unavailable in the same incident. Can Customer claim three separate availability credits?

**Ground truth:** No. When base-platform unavailability also makes dependent modules unavailable, duplicative credits are prohibited. The credit is calculated once using the highest applicable monthly recurring fee among affected services, which is the base platform's fee on the stated price table.

**Reasoning tags:** anti-double-recovery, cross-document

**Required citations:**
- `03_SERVICE_LEVEL_AND_SUPPORT_SCHEDULE.md` → `12. Service Measurement Across Modules`
- `02_ENTERPRISE_ORDER_FORM_AND_CHANGE_ORDERS.md` → `2.1 Recurring Fees`

## LC-08
**Question:** Provider performed six hours of properly noticed Sunday Scheduled Maintenance, and the service was unavailable throughout. How many hours can count as Unavailable for SLA calculation?

**Ground truth:** Two hours can count as Unavailable. Scheduled Maintenance is excluded only up to four hours per month; excess maintenance minutes count as Unavailable when the Covered Service is unavailable.

**Reasoning tags:** cap, formula-input

**Required citations:**
- `03_SERVICE_LEVEL_AND_SUPPORT_SCHEDULE.md` → `3. Excluded Minutes`
- `03_SERVICE_LEVEL_AND_SUPPORT_SCHEDULE.md` → `9. Planned Maintenance`

## LC-09
**Question:** A Priority 1 incident caused downtime and Provider also missed its initial-response target by 45 minutes. May Customer receive both credit types?

**Ground truth:** Yes. Availability and responsiveness credits may both apply to the same incident. Each must be calculated against its own fee base and cap: the affected service's monthly recurring fee for availability, and the monthly Premium Support fee for responsiveness.

**Reasoning tags:** parallel-remedies, separate-caps

**Required citations:**
- `03_SERVICE_LEVEL_AND_SUPPORT_SCHEDULE.md` → `4. Availability Credits`
- `03_SERVICE_LEVEL_AND_SUPPORT_SCHEDULE.md` → `7. Support Responsiveness Credits`

## LC-10
**Question:** Provider confirms unauthorized access to Customer Personal Data at 10:00 Monday but waits until it has complete forensic findings at 16:00 Tuesday to notify Customer. Is the notice timely?

**Ground truth:** No. The DPA requires notice without undue delay and no later than 24 hours after confirmation. It also says an initial notice may be preliminary and must not be delayed for complete information. Notification at 16:00 Tuesday is 30 hours after confirmation.

**Reasoning tags:** deadline, explicit-anti-delay

**Required citations:**
- `04_DATA_PROCESSING_AND_SECURITY_ADDENDUM.md` → `6.1 Notification`
- `04_DATA_PROCESSING_AND_SECURITY_ADDENDUM.md` → `6.2 Contents and Updates`

## LC-11
**Question:** A Security Incident caused by Provider's ordinary DPA breach triggers legally required notifications. Are reasonable notification and identity-protection costs consequential damages barred by the MSA?

**Ground truth:** No. The MSA expressly permits recovery of reasonable breach-notification and identity-protection costs despite the consequential-loss exclusion. The DPA classifies reasonable documented incident costs as direct damages when caused by Provider's breach, subject to the Enhanced Cap absent fraud or willful misconduct.

**Reasoning tags:** exception-to-exclusion, liability-cap

**Required citations:**
- `01_MASTER_SERVICES_AGREEMENT.md` → `9.4 Consequential Loss`
- `04_DATA_PROCESSING_AND_SECURITY_ADDENDUM.md` → `6.3 Costs`
- `04_DATA_PROCESSING_AND_SECURITY_ADDENDUM.md` → `13. Liability and Remedies`

## LC-12
**Question:** Customer objects to a new Subprocessor 18 days after receiving notice. Can it use the DPA's objection-and-termination procedure?

**Ground truth:** No. Customer had 15 calendar days after notice to object. An objection on day 18 is late and constitutes approval under the DPA, so the special objection-based termination route is unavailable on these facts.

**Reasoning tags:** deadline, deemed-approval

**Required citations:**
- `04_DATA_PROCESSING_AND_SECURITY_ADDENDUM.md` → `7. Subprocessors`

## LC-13
**Question:** Provider wants to relocate primary production hosting from India to Singapore based on an account-manager email approving the move. Is the move authorized?

**Ground truth:** No. Moving primary hosting outside India requires Customer's prior written consent through a signed Change Order expressly amending DPA Section 8. An account-manager email cannot amend the agreement, and the DPA has priority for data-location matters.

**Reasoning tags:** formalities, hierarchy, data-location

**Required citations:**
- `04_DATA_PROCESSING_AND_SECURITY_ADDENDUM.md` → `8. Cross-Border Transfers and Data Location`
- `04_DATA_PROCESSING_AND_SECURITY_ADDENDUM.md` → `1. Scope and Priority`
- `01_MASTER_SERVICES_AGREEMENT.md` → `1.3 No Implied Amendment`

## LC-14
**Question:** Customer uploaded health information after support personnel said in a ticket that it was acceptable. Who bears remediation costs when the upload is discovered?

**Ground truth:** The Restricted Data was not authorized because a ticket cannot replace a signed Change Order expressly amending DPA Section 4. Provider must notify Customer and reasonably assist deletion. Customer ordinarily bears remediation costs, unless Provider knowingly encouraged the upload contrary to the DPA. The support statement may be relevant evidence of knowing encouragement, but the facts do not establish the speaker's knowledge or authority, so cost allocation cannot be conclusively determined.

**Reasoning tags:** loophole-analysis, burden, insufficient-facts

**Required citations:**
- `04_DATA_PROCESSING_AND_SECURITY_ADDENDUM.md` → `12. Restricted Data Exception Procedure`
- `01_MASTER_SERVICES_AGREEMENT.md` → `1.3 No Implied Amendment`

## LC-15
**Question:** Customer receives an invoice with no purchase-order number after ignoring Provider's written request for that number for six business days. Can Customer treat it as invalid and avoid the payment clock?

**Ground truth:** No. OF-7 says absence of the purchase-order number does not invalidate the invoice if Customer failed to provide it within five business days after Provider's written request. Assuming the other required fields are present, the invoice is valid and the payment clock runs.

**Reasoning tags:** anti-loophole, condition

**Required citations:**
- `02_ENTERPRISE_ORDER_FORM_AND_CHANGE_ORDERS.md` → `7. Invoice Requirements`
- `01_MASTER_SERVICES_AGREEMENT.md` → `3.2 Invoices and Payment`

## LC-16
**Question:** Customer disputed 20% of a valid invoice with detailed notice on day 12, but withheld the entire invoice. May Provider suspend immediately?

**Ground truth:** No. Customer was entitled to withhold only the disputed 20% and had to pay the undisputed portion. However, Provider still cannot suspend immediately: it must give the required ten-business-day notice and an additional five-business-day cure period, and suspension cannot be based on the properly disputed amount.

**Reasoning tags:** partial-right, procedural-remedy

**Required citations:**
- `01_MASTER_SERVICES_AGREEMENT.md` → `3.3 Good-Faith Disputes`
- `01_MASTER_SERVICES_AGREEMENT.md` → `3.4 Late Payment and Suspension`

## LC-17
**Question:** Provider missed a milestone, then two days later claimed the delay was Customer-Caused and issued the first dependency notice. Does the contractual delay extension apply?

**Ground truth:** No. Provider may not classify a delay as Customer-Caused after the affected milestone has already been missed. It also had to identify the dependency, expected impact, and cure opportunity in writing before relying on the mechanism.

**Reasoning tags:** anti-retroactivity, notice-elements

**Required citations:**
- `01_MASTER_SERVICES_AGREEMENT.md` → `2.3 Customer-Caused Delay`
- `02_ENTERPRISE_ORDER_FORM_AND_CHANGE_ORDERS.md` → `4. Customer Dependencies`

## LC-18
**Question:** Customer discovers a qualifying lower-priced competitor contract but requests most-favored pricing 100 days after receiving reasonably sufficient evidence. Must Provider grant the adjustment?

**Ground truth:** No. Even if the comparator satisfies all substantive similarity tests, OF-7 requires Customer to request review within 90 days after receiving reasonably sufficient evidence. A day-100 request misses that contractual window.

**Reasoning tags:** deadline, conditional-right

**Required citations:**
- `02_ENTERPRISE_ORDER_FORM_AND_CHANGE_ORDERS.md` → `2.4 Most-Favored Pricing`

## LC-19
**Question:** Provider assigns the agreement during an asset sale to Customer's direct competitor and gives notice. What option does Customer have?

**Ground truth:** The ordinary asset-sale assignment exception does not apply because the assignee is a direct competitor of the nonassigning party. In addition, the MSA expressly permits Customer to terminate within 30 days after notice of an assignment to a direct competitor.

**Reasoning tags:** exception, termination-window

**Required citations:**
- `01_MASTER_SERVICES_AGREEMENT.md` → `13. Assignment and Change of Control`

## LC-20
**Question:** The CFO asks whether verbal consent from both CEOs is enough to waive the DPA's India-hosting restriction permanently. What is the ground-truth response?

**Ground truth:** No. Permanent authorization requires a signed Change Order expressly amending DPA Section 8. Verbal consent, even from senior executives, does not satisfy the amendment formalities. The parties would need to execute the required document.

**Reasoning tags:** formalities, anti-waiver

**Required citations:**
- `04_DATA_PROCESSING_AND_SECURITY_ADDENDUM.md` → `8. Cross-Border Transfers and Data Location`
- `04_DATA_PROCESSING_AND_SECURITY_ADDENDUM.md` → `1. Scope and Priority`
- `01_MASTER_SERVICES_AGREEMENT.md` → `1.3 No Implied Amendment`
