.PHONY: doctor package submit

BOT ?= baseline
SUBMIT_FLAGS := $(if $(filter 1,$(ACTIVATE)),--activate,)

doctor:
	python3 scripts/doctor.py

# Build artifacts/$(BOT).zip from bots/$(BOT)/MANIFEST without uploading.
package:
	python3 scripts/submit.py $(BOT) --dry-run

# Upload bots/$(BOT) and wait for validation. ACTIVATE=1 joins rated games automatically.
submit:
	python3 scripts/submit.py $(BOT) $(SUBMIT_FLAGS)
