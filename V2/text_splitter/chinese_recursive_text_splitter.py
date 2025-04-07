

import re
from typing import List, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


def _split_text_with_regex_from_start(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            r"第\S*条",
            r"第\S*编"
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # biexuele
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_start(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip()!=""]


if __name__ == "__main__":
    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=50,
        chunk_overlap=0
    )
    ls = [
        """第一条 为保障煤矿安全生产和从业人员的人身安全与健康，防止煤矿事故与职业病危害，根据《煤炭法》《矿山安全法》《安全生产法》《职业病防治法》《煤矿安全监察条例》和《安全生产许可证条例》等，制定本规程。第二条 在中华人民共和国领域内从事煤炭生产和煤矿建设活动，必须遵守本规程。第四条 从事煤炭生产与煤矿建设的企业(以下统称煤矿企业)必须遵守国家有关安全生产的法律、法规、规章、规程、标准和技术规范。煤矿企业必须加强安全生产管理，建立健全各级负责人、各部门、各岗位安全生产与职业病危害防治责任制。第1编，啊实打实是撒安神是。煤矿企业必须建立健全安全生产与职业病危害防治目标管理、投入、奖惩、技术措施审批、培训、办公会议制度，安全检查制度，事故隐患排查、治理、报告制度，事故报告与责任追究制度等。煤矿企业必须建立各种设备、设施检查维修制度，定期进行检查维修，并做好记录。煤矿必须制定本单位的作业规程和操作规程。第五条 煤矿企业必须设置专门机构负责煤矿安全生产与职业病危害防治管理作，配备满足工作需要的人员及装备。第六条 煤矿建设项目的安全设施和职业病危害防护设施，必须与主体工程同时设计、同时施工、同时投入使用。第七条 对作业场所和工作岗位存在的危险有害因素及防范措施、事故应急措施、职业病危害及其后果、职业病危害防护措施等，煤矿企业应当履行告知义务，从业人员有权了解并提出建议。""",
        ]
    # text = """"""
    for inum, text in enumerate(ls):
        print(inum)
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            print(chunk)
