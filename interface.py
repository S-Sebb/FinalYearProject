# -*- coding: utf-8 -*-
import streamlit as st

from predict_classifier import init_classifier_model_tokenizer, classify_abstract

classifier_model_path = "classifier_model.pt"
classifier_tokenizer_path = "classifier_tokenizer.pt"
classifier_model, classifier_tokenizer = init_classifier_model_tokenizer(classifier_model_path,
                                                                         classifier_tokenizer_path)
ids_to_labels = {0: "OBJECTIVE&BACKGROUND", 1: "METHODS", 2: "RESULTS", 3: "CONCLUSIONS"}

# set page config
st.set_page_config(
    page_title="NER"
)

st.title("Demo")

default_text = """The current objective of antiglaucomatous therapy is to reduce intraocular pressure (IOP), and thus to preserve visual function. Many ophthalmologists believe this objective is best achieved by methods that improve ocular blood flow to the optic nerve head. Beta-blockers are effective ocular hypotensive agents, but they can reduce choroidal blood flow. Bimatoprost, a new prostamide analogue, has been shown to have a better IOP-lowering effect compared with the nonselective beta-adrenergic receptor blocker timolol maleate, but little is known about its effects on the vascular bed of the eye.

The aim of this study was to compare the effects of bimatoprost and timolol on IOP and choroidal blood flow (as measured using pulsatile ocular blood flow [pOBF]) in patients with primary open-angle glaucoma (POAG).

This prospective, open-label, randomized, 2-arm, parallel-group study was conducted at the Glaucoma Research Centre, Department of Ophthalmology, University Hospital of Bari, Bari, Italy. Patients with POAG having well-controlled IOP (<16 mm Hg) on monotherapy with timolol 0.5% ophthalmic solution (2 drops per affected eye BID) for ≥12 months but with a progressive decrease in pOBF during the same time period were randomly allocated to 1 of 2 treatment groups. One group continued monotherapy with timolol, 2 drops per affected eye BID. The other group was switched (without washout) to bimatoprost 0.3% ophthalmic solution (2 drops per affected eye QD [9 pm]). Treatment was given for 180 days. IOP and pOBF were assessed at the diagnostic visit (pre-timolol), baseline (day 0), and treatment days 15, 30, 60, 90, and 180. Primary adverse effects (AEs) (ie, conjunctival hyperemia, conjunctival papillae, stinging, burning, foreign body sensation, and pigmentation of periorbital skin) were monitored throughout the study.

Thirty-eight patients were enrolled (22 men, 16 women; mean [SD] age, 51.7 [4.8] years; 19 patients per treatment group; 38 eligible eyes). At 180-day follow-up in the timolol group, the IOP and the pOBF remained unchanged compared with baseline. In the bimatoprost group the IOP remained unchanged and the pOBF was decreased by 38.9% compared with baseline (P < 0.01). All AEs were mild to moderate and included conjunctival hyperemia and ocular itching (5 patients [26.3%] in the bimatoprost group) and pigmentation of periorbital skin (2 patients [40.0%] in the bimatoprost group). The incidence of each AE was higher in the bimatoprost group than in the timolol group (P = 0.008).

In this population of patients with POAG, bimatoprost was associated with increased pOBF, and the reduction in pOBF associated with timolol was corrected after patients were switched to bimatoprost. Bimatoprost was associated with increased choroidal blood flow, beyond the levels recorded before timolol treatment. The decreased IOP level achieved in the timolol group seemed to be improved further by bimatoprost. Considering the potential efficacy of bimatoprost on IOP and pOBF, we suggest that this new drug may represent a clinical advance in the medical treatment of POAG.
"""

st.subheader("Enter RCT paper abstract for analysis")
text = st.text_area("Paste full RCT paper abstract below", default_text, height=200)

classify_result = classify_abstract(text, classifier_model, classifier_tokenizer, ids_to_labels)
for label, paragraph in classify_result.items():
    st.subheader(label)
    st.write(paragraph)
