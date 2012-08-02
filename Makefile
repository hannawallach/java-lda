BUILD_DIR = build
LIBS_DIR = libs
SRC_DIR = src
DATA_DIR = data
RESULTS_DIR = results
SCRIPTS_DIR = scripts

MAX_HEAP = 1500m

JAVA_FLAGS = -server -enableassertions -Xmx$(MAX_HEAP) -XX:MaxPermSize=500m

CP = $(BUILD_DIR):$(LIBS_DIR)/mallet.jar:$(LIBS_DIR)/mallet-deps.jar

# by default simply compile source code

all: $(BUILD_DIR)

.PHONY: $(BUILD_DIR)

# compilation is handled by ant

$(BUILD_DIR): #clean
	ant build

# experiments...

.PRECIOUS: $(DATA_DIR)/patents/%

$(DATA_DIR)/patents/%: $(DATA_DIR)/patents/%.tar.gz
	tar zxvf $< -C $(@D)

$(DATA_DIR)/patents_%.dat: $(DATA_DIR)/patents/%
	java $(JAVA_FLAGS) \
	-classpath $(CP) \
	cc.mallet.classify.tui.Text2Vectors \
	--keep-sequence \
	--output $@ \
	--input $<

$(DATA_DIR)/patents_%_no_stop.dat: $(DATA_DIR)/patents/%
	java $(JAVA_FLAGS) \
	-classpath $(CP) \
	cc.mallet.classify.tui.Text2Vectors \
	--keep-sequence \
	--remove-stopwords \
	--extra-stopwords $(DATA_DIR)/stopwordlist.txt \
	--output $@ \
	--input $<

$(RESULTS_DIR)/lda/%/T$(T)-S$(S)-SYM$(SYM)-OPT$(OPT)-ID$(ID): $(BUILD_DIR)
	mkdir -p $@; \
	I=`expr $(S) / 10`; \
	java $(JAVA_FLAGS) \
	-classpath $(CP) \
	edu.umass.cs.wallach.Experiment \
	$(DATA_DIR)/$*.dat \
	$(T) \
	$(S) \
	20 \
	$$I \
	$(SYM) \
	$(OPT) \
	$@ \
	> $@/stdout.txt

clean:
	ant clean
