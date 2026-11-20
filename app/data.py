"""Milestone 1 data: legal clauses from public-sector contracts and records, each with one tag.

The built-in corpus is generated from hand-written clause templates with randomised slots (parties,
days, amounts, statutes), so it works offline and is the same on every machine for a given seed.
Each tag has several templates, and the last TEST_TEMPLATES_PER_TAG of them are held out: test
clauses use phrasings the model never saw in training, so test accuracy means generalisation, not
memorisation. You can also bring your own data as JSONL lines: {"text": "...", "tag": "payment"}.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

TAGS = [
    "termination",
    "confidentiality",
    "indemnification",
    "governing_law",
    "payment",
    "limitation_of_liability",
    "public_records",
    "insurance",
    "non_discrimination",
    "force_majeure",
]

TEST_TEMPLATES_PER_TAG = 2

SLOTS = {
    "agency": ["the City", "the County", "the Department", "the Authority", "the Board of Commissioners",
               "the Municipality", "the State Agency", "the School District", "the Transit Authority",
               "the Housing Authority", "the Public Works Department", "the Water District"],
    "vendor": ["the Contractor", "the Vendor", "the Consultant", "the Provider", "the Supplier",
               "the Grantee", "the Service Provider", "the Licensee", "the Engineer", "the Lessee"],
    "days": ["ten (10)", "fifteen (15)", "thirty (30)", "forty-five (45)", "sixty (60)", "ninety (90)",
             "five (5) business", "seven (7)", "twenty (20)", "120"],
    "amount": ["$1,000,000", "$2,000,000", "$500,000", "$5,000,000", "$250,000", "$3,000,000",
               "one million dollars ($1,000,000)", "two million dollars ($2,000,000)"],
    "state": ["California", "Texas", "New York", "Illinois", "Ohio", "Washington", "Georgia",
              "Pennsylvania", "Arizona", "Colorado", "Michigan", "Oregon", "Virginia", "Florida"],
    "court": ["the Superior Court", "the District Court", "the Circuit Court", "the Court of Common Pleas",
              "the state courts", "a court of competent jurisdiction"],
    "records_act": ["the Public Records Act", "the Freedom of Information Act", "the Open Records Act",
                    "the Public Information Act", "the Right-to-Know Law", "the Sunshine Law"],
    "doc": ["this Agreement", "this Contract", "this Ordinance", "this Memorandum of Understanding",
            "this Grant Agreement", "this Lease", "this Purchase Order", "this Interlocal Agreement"],
    "pct": ["one and one-half percent (1.5%)", "one percent (1%)", "two percent (2%)", "0.5%", "ten percent (10%)"],
}

# Ten templates per tag. The last TEST_TEMPLATES_PER_TAG of each list are the held-out test phrasings.
TEMPLATES: dict[str, list[str]] = {
    "termination": [
        "{agency} may terminate {doc} for convenience upon {days} days' written notice to {vendor}.",
        "Either party may terminate {doc} if the other party materially breaches it and fails to cure the breach within {days} days after written notice.",
        "In the event funds are not appropriated for any fiscal year, {agency} may terminate {doc} without penalty effective at the end of the last funded period.",
        "Upon termination, {vendor} shall immediately cease all work, deliver all work product to {agency}, and be paid only for services satisfactorily performed through the termination date.",
        "{doc} shall remain in effect until terminated by {agency}, which may end it at any time by giving {vendor} {days} days' notice.",
        "{agency} reserves the right to cancel {doc}, in whole or in part, immediately if {vendor} becomes insolvent or files for bankruptcy.",
        "Termination of {doc} for cause shall not relieve {vendor} of liability to {agency} for damages sustained by virtue of any breach.",
        "If {vendor} defaults on any obligation, {agency} may, after {days} days' notice and opportunity to cure, end {doc} and procure the services elsewhere.",
        "Should {vendor} fail to perform, {agency} may discontinue the engagement under {doc} by written notice, and the engagement shall conclude on the date specified in that notice.",
        "{doc} may be ended early by mutual written consent, or unilaterally by {agency} upon {days} days' advance notice, with {vendor} compensated for work completed to that date.",
    ],
    "confidentiality": [
        "{vendor} shall hold in strict confidence all confidential information disclosed by {agency} and shall not disclose it to any third party without prior written consent.",
        "{vendor} shall use confidential information solely to perform its obligations under {doc} and shall protect it with at least reasonable care.",
        "All personally identifiable information of residents obtained under {doc} shall be kept confidential and protected against unauthorized access or disclosure.",
        "{vendor} shall ensure that its employees and subcontractors with access to confidential information are bound by written nondisclosure obligations at least as protective as these.",
        "Upon expiration of {doc}, {vendor} shall return or securely destroy all confidential information of {agency} and certify the destruction in writing.",
        "The obligations of non-disclosure in this section shall survive the termination of {doc} for a period of five years.",
        "{vendor} shall promptly notify {agency}, within {days} days, of any unauthorized use or disclosure of confidential data and cooperate to mitigate its effects.",
        "Confidential information does not include information that is publicly available, already known to {vendor}, or independently developed without use of the disclosing party's materials.",
        "{vendor} agrees not to reveal, publish, or otherwise make known any sensitive data, trade secrets, or security plans of {agency} learned while performing {doc}.",
        "Any non-public material furnished by {agency} shall be treated as proprietary, kept secure, and shared only with personnel who need it to carry out {doc}.",
    ],
    "indemnification": [
        "{vendor} shall indemnify, defend, and hold harmless {agency}, its officers, and employees from and against all claims, damages, losses, and expenses arising out of {vendor}'s performance of {doc}.",
        "To the fullest extent permitted by law, {vendor} shall defend and indemnify {agency} against any third-party claim caused by the negligent acts or omissions of {vendor} or its subcontractors.",
        "{vendor} shall indemnify {agency} against any claim that the deliverables infringe any patent, copyright, or trade secret of a third party.",
        "The duty to defend under this section arises immediately upon presentation of a claim, and {vendor} shall bear all attorneys' fees and costs of defense.",
        "{vendor}'s obligation to hold {agency} harmless shall not apply to the extent a loss is caused by the sole negligence or willful misconduct of {agency}.",
        "{agency} shall give {vendor} prompt notice of any claim for which indemnity is sought, and {vendor} shall control the defense with counsel reasonably acceptable to {agency}.",
        "The indemnification obligations of {vendor} shall survive the expiration or termination of {doc}.",
        "{vendor} shall reimburse {agency} for any fines, penalties, or judgments imposed as a result of {vendor}'s violation of applicable law.",
        "{vendor} agrees to protect {agency} and its elected officials from, and compensate them for, any lawsuit, demand, or liability resulting from {vendor}'s work under {doc}.",
        "Should any person bring an action against {agency} because of acts of {vendor}, {vendor} will assume the defense of that action at its own expense and pay any resulting award.",
    ],
    "governing_law": [
        "{doc} shall be governed by and construed in accordance with the laws of the State of {state}, without regard to its conflict of laws principles.",
        "Any action arising out of {doc} shall be brought exclusively in {court} located in the county where {agency} is situated.",
        "The parties consent to the jurisdiction of {court} of {state} and waive any objection to venue there.",
        "{doc} is entered into in {state} and its interpretation and enforcement shall be controlled by {state} law.",
        "Venue for any dispute relating to {doc} shall lie in {court}, and the parties waive any right to a jury trial to the extent permitted by law.",
        "Nothing in {doc} shall be construed as a waiver of the sovereign immunity of {agency} under the laws of {state}.",
        "Disputes not resolved by negotiation shall be submitted to {court}, applying the substantive law of {state}.",
        "The validity, performance, and effect of {doc} shall be determined under the statutes and case law of {state}.",
        "All questions concerning the meaning of {doc} will be decided under {state} law, and any suit must be filed in {court}.",
        "The parties agree that {state} law controls every aspect of {doc} and that {court} shall be the sole forum for litigation.",
    ],
    "payment": [
        "{agency} shall pay {vendor} within {days} days after receipt of a proper invoice for services accepted under {doc}.",
        "{vendor} shall submit monthly invoices itemizing the services performed, hours worked, and expenses incurred, together with supporting documentation.",
        "The total compensation payable under {doc} shall not exceed {amount} without a written amendment approved by {agency}.",
        "Late payments shall accrue interest at {pct} per month as provided by the prompt payment statute.",
        "{agency} may withhold payment of any disputed amount until the dispute is resolved, and shall pay the undisputed portion when due.",
        "{agency} may retain ten percent (10%) of each progress payment until final completion and acceptance of the work.",
        "Travel expenses shall be reimbursed only at the rates established by {agency}'s travel policy and only when approved in advance.",
        "Payment shall be made by electronic funds transfer to the account designated by {vendor} in writing.",
        "{vendor} will be compensated at the fixed fee set out in Exhibit B, billed in arrears, with {agency} remitting each amount owed within {days} days.",
        "The price for the goods is {amount}, which {agency} will disburse in installments as each milestone is accepted and an invoice is received.",
    ],
    "limitation_of_liability": [
        "In no event shall either party be liable for any indirect, incidental, consequential, special, or punitive damages arising out of {doc}.",
        "{vendor}'s total liability under {doc} shall not exceed the total amount paid by {agency} during the twelve months preceding the claim.",
        "The limitations of liability in this section shall not apply to claims for bodily injury, death, or damage to real property, or to {vendor}'s indemnification obligations.",
        "Neither party shall be liable for lost profits, lost revenue, or loss of data, even if advised of the possibility of such damages.",
        "The aggregate liability of {vendor} for all claims arising under {doc} is capped at {amount}.",
        "{agency}'s liability under {doc} is limited to the amount of the unpaid contract price and shall in no case exceed the funds appropriated.",
        "These exclusions of damages apply regardless of the form of action, whether in contract, tort, strict liability, or otherwise.",
        "Nothing in {doc} limits liability for fraud, gross negligence, or willful misconduct.",
        "Under no circumstances will {agency} be responsible for any amount beyond {amount}, and remote or speculative losses are excluded entirely.",
        "Recovery by {vendor} for any breach by {agency} is restricted to direct damages, and the maximum exposure of either party is the contract value.",
    ],
    "public_records": [
        "{vendor} acknowledges that {agency} is subject to {records_act} and that {doc} and related documents may be disclosed to the public upon request.",
        "If {agency} receives a request under {records_act} for records held by {vendor}, {vendor} shall produce the records to {agency} within {days} days.",
        "{vendor} shall clearly mark any material it considers a trade secret exempt from disclosure under {records_act}; unmarked materials will be treated as public records.",
        "{vendor} shall keep and maintain public records required by {agency} to perform the service and shall provide the public with access on the same terms as {agency}.",
        "{agency} shall notify {vendor} of any request for records designated as exempt, and {vendor} may seek a protective order at its own expense.",
        "{vendor} shall retain all records relating to {doc} for at least seven years after final payment and make them available for audit and public inspection.",
        "Upon completion of {doc}, {vendor} shall transfer to {agency} all public records in its possession in a format compatible with {agency}'s information systems.",
        "{agency} shall not be liable for the release of any document it determines in good faith is required to be disclosed by law.",
        "Documents created under {doc} belong to {agency} and may be released to any member of the public who asks for them, consistent with {records_act}.",
        "Because {agency} is a government body, correspondence and files exchanged under {doc} are open to citizen inspection unless a statutory exemption applies.",
    ],
    "insurance": [
        "{vendor} shall maintain commercial general liability insurance with limits of not less than {amount} per occurrence for the duration of {doc}.",
        "{vendor} shall carry workers' compensation insurance as required by the laws of {state} and employer's liability insurance with limits of at least {amount}.",
        "{agency} shall be named as an additional insured on {vendor}'s general liability and automobile liability policies.",
        "{vendor} shall furnish certificates of insurance to {agency} prior to commencing work and shall provide {days} days' notice of any cancellation of coverage.",
        "{vendor} shall maintain professional liability (errors and omissions) insurance of not less than {amount} per claim for three years after completion.",
        "All policies shall be issued by insurers authorized to do business in {state} with an A.M. Best rating of A- or better.",
        "{vendor}'s insurance shall be primary and non-contributory with respect to any insurance or self-insurance maintained by {agency}.",
        "{vendor} shall require each subcontractor to maintain insurance coverage meeting the same requirements as {vendor}.",
        "Throughout the term, {vendor} must keep in force a cyber liability policy of {amount} and a business auto policy, and deliver proof of coverage on request.",
        "{vendor} will obtain and keep active the bonds and policies of coverage listed in Exhibit C, each endorsed to protect {agency}, before any work begins.",
    ],
    "non_discrimination": [
        "{vendor} shall not discriminate against any employee or applicant for employment because of race, color, religion, sex, national origin, age, or disability.",
        "{vendor} shall comply with Title VI of the Civil Rights Act and shall not exclude any person from participation in the program on the basis of race, color, or national origin.",
        "{vendor} shall provide services under {doc} in a manner accessible to persons with disabilities as required by the Americans with Disabilities Act.",
        "{vendor} shall post notices setting forth the provisions of this equal opportunity clause in conspicuous places available to employees and applicants.",
        "{vendor} shall include the provisions of this nondiscrimination section in every subcontract so that they are binding on each subcontractor.",
        "{vendor} shall take affirmative action to ensure that applicants are employed without regard to sexual orientation, gender identity, or veteran status.",
        "Any violation of this equal employment requirement shall be a material breach of {doc} and may result in suspension or debarment.",
        "{vendor} shall make reasonable efforts to use minority-owned and women-owned business enterprises as subcontractors and report participation to {agency}.",
        "{vendor} commits to treat all workers and members of the public fairly, without bias based on ethnicity, creed, gender, or handicap, in every activity under {doc}.",
        "Hiring, promotion, and pay decisions by {vendor} must be made on merit alone, and no person may be denied benefits of {doc} because of a protected characteristic.",
    ],
    "force_majeure": [
        "Neither party shall be liable for any failure or delay in performance caused by events beyond its reasonable control, including acts of God, fire, flood, war, terrorism, or epidemic.",
        "A party affected by a force majeure event shall notify the other party within {days} days and use diligent efforts to resume performance.",
        "Performance obligations shall be suspended for the duration of the force majeure event, and the schedule shall be equitably extended.",
        "Force majeure shall not include economic hardship, changes in market conditions, or {vendor}'s inability to obtain financing.",
        "If a force majeure event continues for more than {days} days, either party may terminate {doc} by written notice without liability.",
        "Strikes, labor disputes, government orders, and pandemics shall constitute excusable delays to the extent they prevent performance.",
        "{vendor} shall have a disaster recovery plan, and the existence of a force majeure event shall not excuse {vendor}'s failure to implement that plan.",
        "Delays caused by severe weather, earthquakes, or utility outages shall not constitute a default under {doc}.",
        "No party will be held in breach where a hurricane, riot, embargo, or other unforeseeable catastrophe makes performance of {doc} impossible for a time.",
        "Should extraordinary circumstances such as a public health emergency or natural disaster prevent {vendor} from working, its deadlines are excused for as long as those circumstances last.",
    ],
}


def _fill(template: str, rng: random.Random) -> str:
    values = {name: rng.choice(options) for name, options in SLOTS.items()}
    text = template.format(**values)
    return text[0].upper() + text[1:]


def generate_clauses(per_template: int = 12, seed: int = 0) -> dict[str, list[dict]]:
    """Generate the built-in corpus, split into train/val/test.

    Train and val come from the first templates of each tag (val is a random 15% slice); test comes only
    from the held-out templates.
    """
    rng = random.Random(seed)
    train_pool, test = [], []
    for tag in TAGS:
        templates = TEMPLATES[tag]
        cut = len(templates) - TEST_TEMPLATES_PER_TAG
        for i, template in enumerate(templates):
            seen = set()
            for _ in range(per_template * 4):  # extra tries so duplicate fills can be skipped
                text = _fill(template, rng)
                if text in seen:
                    continue
                seen.add(text)
                (train_pool if i < cut else test).append({"text": text, "tag": tag, "template": f"{tag}/{i}"})
                if len(seen) == per_template:
                    break
    rng.shuffle(train_pool)
    n_val = int(len(train_pool) * 0.15)
    return {"train": train_pool[n_val:], "val": train_pool[:n_val], "test": test}


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if "text" not in row or "tag" not in row:
                raise ValueError(f"each line needs 'text' and 'tag': {line[:80]}")
            rows.append(row)
    return rows


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
