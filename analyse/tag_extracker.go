// Package analyse is the Golang implementation of Jieba's analyse module.
package analyse

import (
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/kricen/jiebago"
)

// Segment represents a word with weight.
type Segment struct {
	text   string
	weight float64
}

// Text returns the segment's text.
func (s Segment) Text() string {
	return s.text
}

// Weight returns the segment's weight.
func (s Segment) Weight() float64 {
	return s.weight
}

// Segments represents a slice of Segment.
type Segments []Segment

func (ss Segments) Len() int {
	return len(ss)
}

func (ss Segments) Less(i, j int) bool {
	if ss[i].weight == ss[j].weight {
		return ss[i].text < ss[j].text
	}

	return ss[i].weight < ss[j].weight
}

func (ss Segments) Swap(i, j int) {
	ss[i], ss[j] = ss[j], ss[i]
}

// TagExtracter is used to extract tags from sentence.
type TagExtracter struct {
	seg      *jiebago.Segmenter
	idf      *Idf
	stopWord *StopWord
}

// LoadDictionary reads the given filename and create a new dictionary.
func (t *TagExtracter) LoadDictionary(fileName string) error {
	t.stopWord = NewStopWord()
	t.seg = new(jiebago.Segmenter)
	return t.seg.LoadDictionary(fileName)
}

func (t *TagExtracter) GetSegmenter() *jiebago.Segmenter {
	return t.seg
}

func (t *TagExtracter) GetStopWord() *StopWord {
	return t.stopWord
}

// LoadIdf reads the given file and create a new Idf dictionary.
func (t *TagExtracter) LoadIdf(fileName string) error {
	t.idf = NewIdf()
	return t.idf.loadDictionary(fileName)
}

// LoadStopWords reads the given file and create a new StopWord dictionary.
func (t *TagExtracter) LoadStopWords(fileName string) error {
	t.stopWord = NewStopWord()
	return t.stopWord.loadDictionary(fileName)
}

// ExtractTags extracts the topK key words from sentence.
func (t *TagExtracter) ExtractTags(sentence string, topK int) (tags Segments) {
	freqMap := make(map[string]float64)

	for w := range t.seg.Cut(sentence, true) {
		w = strings.TrimSpace(w)
		if utf8.RuneCountInString(w) < 2 {
			continue
		}
		if t.stopWord.IsStopWord(w) {
			continue
		}
		if f, ok := freqMap[w]; ok {
			freqMap[w] = f + 1.0
		} else {
			freqMap[w] = 1.0
		}
	}
	total := 0.0
	for _, freq := range freqMap {
		total += freq
	}
	for k, v := range freqMap {
		freqMap[k] = v / total
	}
	ws := make(Segments, 0)
	var s Segment
	for k, v := range freqMap {
		if freq, ok := t.idf.Frequency(k); ok {
			s = Segment{text: k, weight: freq * v}
		} else {
			s = Segment{text: k, weight: t.idf.median * v}
		}
		ws = append(ws, s)
	}
	sort.Sort(sort.Reverse(ws))
	if topK >= 0 && len(ws) > topK {
		tags = ws[:topK]
	} else {
		tags = ws
	}
	return tags
}

func (t *TagExtracter) isDigit(w string) bool {
	if w == "" {
		return false
	}
	for _, r := range w {
		if !unicode.IsNumber(r) {
			return false
		}
	}
	return true
}

func (t *TagExtracter) isPureDigitLetters(str string) bool {
	if str == "" {
		return false
	}
	d, l := false, false
	for _, r := range str {
		if unicode.IsNumber(r) {
			d = true
			continue
		}
		if (r >= 'A' && r <= 'A'+25) || (r >= 'a' && r <= 'a'+25) {
			l = true
			continue
		}
		return false
	}
	if d && l {
		return true
	}
	return false
}

// CnExtractTags extracts the topK key words from sentence.
func (t *TagExtracter) CNExtractTags(sentence string, topK int) (tags Segments, words []string) {
	freqMap := make(map[string]float64)

	numCount := 0
	for w := range t.seg.Cut(sentence, true) {
		w = strings.TrimSpace(w)
		if utf8.RuneCountInString(w) < 2 {
			continue
		}
		if t.stopWord.IsStopWord(w) {
			continue
		}

		if t.isDigit(w) {
			numCount++
			if numCount > 0 {
				continue
			}
		}

		words = append(words, w)
		if _, ok := freqMap[w]; ok {
			freqMap[w] = 1.0
		} else {
			freqMap[w] = 1.0
		}
	}
	/*
		total := 0.0
		for _, freq := range freqMap {
			total += freq
		}
		for k, v := range freqMap {
			freqMap[k] = v / total
		}
	*/
	ws := make(Segments, 0)
	var s Segment
	for k, v := range freqMap {
		if freq, ok := t.idf.Frequency(k); ok {
			s = Segment{text: k, weight: freq * v}
		} else {
			continue
		}
		ws = append(ws, s)
	}
	sort.Sort(sort.Reverse(ws))
	if topK >= 0 && len(ws) > topK {
		tags = ws[:topK]
	} else {
		tags = ws
	}
	return tags, words
}
