# Enterprise Order Form OF-7 and Change Order CO-2

**Customer:** BlueHarbor Retail Limited  
**Provider:** Alderstone Analytics Private Limited  
**Order Form ID:** OF-7  
**Order Effective Date:** 1 February 2026  
**Initial Term:** 36 months, ending 31 January 2029  
**Related MSA:** MSA-AB-2026

> This synthetic document intentionally contains specific terms, exceptions, and a later Change Order. It is designed to test hierarchy, temporal reasoning, calculations, and detection of apparent loopholes.

# Part A: Executed Order Form OF-7

## 1. Purchased Services

Customer purchases:

- Enterprise Analytics Cloud for up to 2,000 Named Users.
- Demand Forecasting Module.
- Inventory Optimization Module.
- Premium Support.
- Implementation and migration services.

A "Named User" is an individual assigned unique credentials. Credentials may not be shared. A suspended employee account does not count toward the limit after Customer disables access. Service accounts used only for machine-to-machine integration do not count, up to 50 accounts.

## 2. Fees

### 2.1 Recurring Fees

| Item | Annual Fee |
|---|---:|
| Enterprise Analytics Cloud | ₹1,80,00,000 |
| Demand Forecasting Module | ₹45,00,000 |
| Inventory Optimization Module | ₹30,00,000 |
| Premium Support | ₹18,00,000 |
| **Total annual recurring fees** | **₹2,73,00,000** |

Recurring fees are invoiced annually in advance.

### 2.2 Implementation Fees

Implementation is fixed at ₹60,00,000, payable 30% at kickoff, 40% after Data Migration Acceptance, and 30% after Production Acceptance.

### 2.3 Usage Overage

Named Users above 2,000 are billed at ₹900 per active excess user per month. An excess user active for any part of a month counts for that month. The first 25 excess users in any month are waived as a buffer; charges begin with the 26th excess user.

### 2.4 Most-Favored Pricing

If Provider enters a new agreement during the Initial Term with a India-based retail customer purchasing all three core modules, at least 2,000 users, a minimum three-year commitment, and substantially similar support and payment terms, at a recurring per-user effective price more than 10% below Customer's effective price, Provider must offer a prospective price adjustment to the qualifying lower price beginning the next contract year.

Promotional pilots, distressed-customer settlements, public-sector procurements, affiliate arrangements, and agreements involving materially different implementation scope or data volumes are excluded. Customer must request review within 90 days after receiving reasonably sufficient evidence. Provider need not disclose another customer's identity or confidential agreement.

## 3. Project Milestones and Acceptance

| Milestone | Target Date | Acceptance Period |
|---|---|---:|
| Design Completion | 31 March 2026 | 10 business days |
| Data Migration Acceptance | 30 June 2026 | 15 business days |
| User Acceptance Testing | 31 August 2026 | 15 business days |
| Production Acceptance | 30 September 2026 | 20 business days |

Customer must reject a milestone in writing within the applicable Acceptance Period, identifying reproducible material nonconformities against the signed acceptance criteria. If Customer neither accepts nor validly rejects within the period, the milestone is deemed accepted.

Use in production does not itself constitute acceptance during an agreed pilot, emergency continuity test, or parallel run. Outside those exceptions, productive use by more than 100 users for ten consecutive business days constitutes acceptance.

Provider must correct valid nonconformities and resubmit. The new review is limited to corrected items and reasonably affected functions; Customer may not reopen previously accepted unrelated items.

## 4. Customer Dependencies

Customer will provide source-system credentials by 15 February 2026, mapped data dictionaries by 1 March 2026, and at least 20 trained UAT participants by 1 August 2026. Provider may claim Customer-Caused Delay only in compliance with MSA Section 2.3.

If a Customer-Caused Delay exceeds 20 business days in aggregate, Provider may charge documented idle-resource costs capped at ₹8,00,000 per month, but only after giving a cost estimate and allowing Customer five business days to avoid the cost by rescheduling.

## 5. Special Renewal and Termination Terms

### 5.1 Renewal

After the Initial Term, OF-7 renews for one 12-month period unless either party provides at least 90 calendar days' notice before 31 January 2029. After that first renewal, further renewals follow the MSA's default 60-day notice rule.

### 5.2 Convenience Termination

Customer may terminate the Demand Forecasting Module for convenience after the first 18 months by giving 120 days' written notice. No refund is due for the current prepaid contract year, but the module will not be invoiced in a contract year beginning after the effective termination date.

Customer has no general convenience termination right for the base platform or Inventory Optimization Module.

### 5.3 Provider Breach

Termination for Provider breach follows the MSA and SLS. A valid termination of the base platform automatically terminates dependent modules without early termination fees.

### 5.4 Acceleration

If Provider terminates OF-7 for Customer's uncured material breach, 70% of remaining recurring fees for the then-current contract year become immediately due. Fees for later contract years do not accelerate.

## 6. Deliverables and Intellectual Property

The Store Replenishment Rules Engine described in Appendix 1 is expressly designated **Customer-Owned Custom Work**. Ownership transfers after Customer pays all implementation fees attributable to it, listed as ₹18,00,000.

The generic ETL framework, forecasting libraries, model-training pipeline, monitoring templates, reusable connectors, and Provider know-how are Provider Background Materials even if modified during implementation. Customer receives the licenses described in the MSA.

## 7. Invoice Requirements

A valid invoice must state the Order Form ID, purchase-order number supplied by Customer, billing period, service line items, applicable taxes, and bank details. Absence of the purchase-order number does not invalidate an invoice if Customer failed to provide the number within five business days after Provider's written request.

## 8. Legal Notice Addresses

Provider legal notices: legal@alderstone.example  
Customer legal notices: legal.notices@blueharbor.example

A copy to an account manager is optional and has no effect on validity.

## 9. Additional Use Restriction

Customer may use service outputs to support internal merchandising, pricing, procurement, and inventory decisions. Customer may not resell raw service access or expose Provider's user interface as a standalone service to third parties. Customer may share static reports with suppliers where reports do not reveal Provider Confidential Information.

# Appendix 1: Store Replenishment Rules Engine

The custom work includes Customer-specific rule definitions, approval flows, and store-cluster mappings. It excludes Provider's generic rule compiler, execution runtime, reusable UI components, testing harness, and deployment tooling.

# Part B: Signed Change Order CO-2

**Change Order ID:** CO-2  
**Effective Date:** 1 November 2026  
**Expressly amends:** OF-7 Sections 2.3, 3, and 6 only

## CO-2.1 Revised User Overage

OF-7 Section 2.3 is replaced as follows:

Named Users above 2,000 are billed at ₹800 per active excess user per month. The first **10** excess users in any month are waived; charges begin with the 11th excess user. Service accounts remain excluded up to 50 accounts. If there are more than 50 service accounts, each account above 50 is treated as a Named User for overage calculations.

## CO-2.2 Revised Production Acceptance

The Production Acceptance target moves to 30 November 2026, with a 20-business-day Acceptance Period. Productive use constitutes acceptance only when at least 250 Named Users use the production service on each of 15 consecutive business days. The pilot, emergency continuity test, and parallel-run exceptions remain unchanged.

This amendment does not reopen Design Completion, Data Migration Acceptance, or User Acceptance Testing milestones.

## CO-2.3 Custom Work Payment

The Store Replenishment Rules Engine remains Customer-Owned Custom Work, but ownership transfers when Customer has paid the **₹18,00,000 specifically allocated amount**, even if another unrelated implementation invoice is disputed in good faith. Provider's Background Materials remain excluded.

## CO-2.4 No Other Amendment

CO-2 changes only the three subjects expressly stated above. All other OF-7, SLS, DPA, and MSA provisions remain unchanged. In particular, CO-2 does not modify service levels, renewal notice, limitation of liability, security obligations, or termination rights.
