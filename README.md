# Convolutional encoder/decoder
Simple convolutional encoder/decoder using base rate of 1/2.

## How to run

`python3 bms.py -e <<< input=[ASCII characters [a-z][A-Z][0-9]] (encoding)`<br>
`python3 bms.py -d <<< input=[bit values of characters [0,1]] (decoding)`

## Examples

`$ python3 bms.py -e <<< test`<br>
`00101100111001101000001010010010001101100000101001000001001001101010110000`
<br>
<br>
`$ python3 bms.py -d <<< 00101100111001101000001010010010001101100000101001000001001001101010110000`<br>
`test`

## Notes

* You can specify any simple convolutional coder with paramameter `-p X Y Z`
	* `X` - number of delay cells (e.g. `5`)
	* `Y` - upper feedback scheme (e.g. `46`=101110)
	* `Z` - lower feedback scheme (e.g. `53`=110101)
* Whitespace and special characters are ignored.
* Initialized 0s are part of output/input.
