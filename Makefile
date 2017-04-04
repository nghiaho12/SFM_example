.PHONY: clean All

All:
	@echo "----------Building project:[ SFM_example - Debug ]----------"
	@"$(MAKE)" -f  "SFM_example.mk"
clean:
	@echo "----------Cleaning project:[ SFM_example - Debug ]----------"
	@"$(MAKE)" -f  "SFM_example.mk" clean
