import datetime
import os
import pickle
import re
import requests
import shutil
from cryptography.fernet import Fernet
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from math import pi as Pi
import subprocess


def Path(*args, reverse=False, check_too=None):
    ONEDRIVE = os.environ.get('OneDrive')
    APPDATA = os.environ.get('AppData')
    PROGRAMFILES = os.environ.get('ProgramFiles')
    paths = {
        'rm_money': os.path.join(ONEDRIVE, r'Documents\Rainmeter\Skins\Money'),
        'plutus': os.path.join(ONEDRIVE, r'Pandora\plutus'),
        'shellnew': "C:\\Windows\\ShellNew",
        'sync_storage': os.path.join(ONEDRIVE, r'Pandora\windows\sync_files'),
        'sync': os.path.join(ONEDRIVE, r'Pandora\windows\sync_files'),
        'themisDir': os.path.join(ONEDRIVE, r'Pandora\windows'),
        'cwd': '...',
        'working dir': '...',
        'appdata_roaming': APPDATA,
        'appdata': APPDATA[:-8],
        'pandora': os.path.join(ONEDRIVE, r'Pandora'),
        'OD': ONEDRIVE,
        'od': ONEDRIVE,
        'onedrive': ONEDRIVE,
        "program_files": PROGRAMFILES,
        # 'key': os.environ.get('KEY'),
        # 'KEY': os.environ.get('KEY'),
        'startup': os.path.join(APPDATA, r'Microsoft\Windows\Start Menu\Programs\Startup'),
        'start_menu': os.path.join(APPDATA, r'Microsoft\Windows\Start Menu'),
        'vlc': os.path.join(PROGRAMFILES, r'VideoLAN\VLC\vlc.exe')
    }

    if os.environ['COMPUTERNAME'] == 'PHYAROM2':
        paths["lyx_bin"] = "C:\\Program Files (x86)\\LyX 2.3"
    else:
        paths["lyx_bin"] = "C:\\Program Files\\LyX 2.3"

    if args in ['computer', 'machine']:
        return os.environ['COMPUTERNAME']

    if check_too is not None:
        for key in check_too:
            paths[key[0]] = key[1]

    if len(args) == 0:
        args = [paths['cwd']]
    elif isinstance(args, str):
        args = [args]
    elif isinstance(args, tuple):
        args = list(args)
    else:
        raise SyntaxError
    for i, key in enumerate(args):
        if key in paths:
            args[i] = f'${key}$'

    if not (('$' in args[0]) or (':\\' in args[0])):
        args = [paths['cwd']] + args

    if reverse:
        match = ['', None]
        for i, term in enumerate(args):
            for key in paths:
                tag = paths[key]
                if (tag in term) and (len(tag) >= len(match[0])):
                    match[0] = tag
                    match[-1] = f'${key}$'
            args[i] = term.replace(*match)

    else:
        for key in paths:
            tag = f'${key}$'
            fold = paths[key]
            for i, term in enumerate(args):
                if tag in term:
                    args[i] = args[i].replace(tag, fold)
                    return os.path.join(*args)

    return os.path.join(*args)


def copy_file(src_path, dst_path="", add_time_stamp=True, backup=True, remove_original=False):
    just_backup = False
    if not os.path.exists(src_path):
        return 'no file'

    if dst_path == '':
        dst_name = os.path.basename(src_path)
        just_backup = True
    else:
        dst_name = os.path.basename(dst_path)

    if os.path.dirname(dst_path) == '':
        dst_folder = os.path.join(os.path.dirname(src_path))
    else:
        dst_folder = os.path.join(os.path.dirname(dst_path))

    dst_path = os.path.join(dst_folder, dst_name)

    if backup:
        if not isinstance(backup, str):
            backup = os.path.join(dst_folder, 'backup')
        elif backup == 'same':
            backup = dst_folder

        if os.path.exists(dst_path):
            name, ext = os.path.splitext(dst_name)
            if add_time_stamp:
                name = f'{name}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")}'
                dst_name = name + ext
            i = 0
            while os.path.exists(os.path.join(backup, dst_name)):
                i += 1
                instance = f' ({i})'
                dst_name = f"{name}{instance}{ext}"

            backup_dst = os.path.join(backup, dst_name)
            os.makedirs(backup, exist_ok=True)
            shutil.copyfile(dst_path, backup_dst)

    if not just_backup:
        os.makedirs(dst_folder, exist_ok=True)
        if not remove_original:
            shutil.copyfile(src_path, dst_path)
        else:
            shutil.move(src_path, dst_path)

    return dst_path


def save(obj, path, store=True):
    if os.path.splitext(path)[-1] == '':
        path += '.pkl'
    path = Path(path)

    if os.path.exists(path) and store:
        copy_file(path, backup=True)

    with open(path, 'wb') as file:
        file.truncate(0)
        pickle.dump(obj, file)
    return path


def load(path):
    if os.path.splitext(path)[-1] == '':
        path += '.pkl'
    path = Path(path)

    with open(path, 'rb') as file:
        return pickle.load(file)


def encrypt(file=None, new_key=False):
    if new_key:
        key = Fernet.generate_key()
        with open(Path('key'), 'wb') as file_key:
            file_key.write(key)

    else:
        with open(Path('KEY'), 'rb') as file_key:
            key0 = file_key.read()
            fernet = Fernet(key0)

        with open(os.path.join(Path('pandora'), r'keys\key.key'), 'rb') as file_key:
            key = fernet.decrypt(file_key.read())

    fernet = Fernet(key)

    with open(file, 'rb') as f:
        original = f.read()
        encrypted = fernet.encrypt(original)
    with open(file, 'wb') as encrypted_file:
        encrypted_file.write(encrypted)


def decrypt(file, save_file=False):
    keys = {'local': Path('KEY'), 'online': os.path.join(Path('pandora'), r'keys\key.key')}

    with open(keys['local'], 'rb') as file_key:
        key0 = file_key.read()
    with open(keys['online'], 'rb') as file_key:
        key = Fernet(key0).decrypt(file_key.read())

    with open(file, 'rb') as f:
        original = f.read()
    decrypted = Fernet(key).decrypt(original)

    if save_file:
        with open(file, 'wb') as f:
            f.write(decrypted)
    else:
        return decrypted


def fix_ext(file):
    file, ext = os.path.splitext(file)
    if ext == '':
        file += '.pkl'
    else:
        file += ext
    return file


def list_options(lst):
    txt = ''
    for i, o in enumerate(lst):
        txt += '({}): {}. '.format(i, o)
    return txt


def cprint(txt=None, c=None, bg=None, styles=None, p=False, get_style=False):
    os.system('color')
    styles_dict = {'b': '1;',
                   'B': '1;',
                   'l': '2;',
                   'L': '2;',
                   'light': '2;',
                   'i': '3;',
                   'I': '3;',
                   'italic': '3;',
                   'u': '4;',
                   'U': '4;',
                   'uline': '4;',
                   'underline': '4;',
                   'bk': '5;',
                   'bl': '5;',
                   'blink': '5;',
                   'blinking': '5;'
                   }
    colors = {'black': '30;',
              'bk': '30;',
              'k': '30;',
              'red': '31;',
              'r': '31;',
              'green': '32;',
              'g': '32;',
              'orange': '33;',
              'o': '33;',
              'blue': '34;',
              'b': '34;',
              'bl': '34;',
              'magenta': '35;',
              'm': '35;',
              'purple': '35;',
              'p': '35;',
              'cyan': '36;',
              'cy': '36;',
              'lightgrey': '37;',
              'darkgrey': '90;',
              'lightred': '91;',
              'lightgreen': '92;',
              'yellow': '93;',
              'y': '93;',
              'lightblue': '94;',
              'pink': '95;',
              'pk': '95;',
              'lightcyan': '96;'}
    backgrounds = {'black': '40;',
                   'bk': '40;',
                   'k': '40;',
                   'red': '41;',
                   'r': '41;',
                   'green': '42;',
                   'g': '42;',
                   'yellow': '43;',
                   'y': '43;',
                   'blue': '44;',
                   'bl': '44;',
                   'b': '44;',
                   'magenta': '45;',
                   'm': '45;',
                   'purple': '45;',
                   'p': '45;',
                   'cyan': '46;',
                   'cy': '46;',
                   'white': '47;',
                   'w': '47;',

                   }

    if txt is None:
        if c == 'rand' or c == 'random':
            return list(colors.keys())[randint(0, len(colors) - 1)]
        else:
            return

    if get_style:
        txt = txt[5:txt.find('m')]
        txt = [ch + ';' for ch in txt.split(';')]
        style = {'c': None, 'bg': None, 'styles': []}
        for key, value in styles_dict.items():
            if value in txt:
                style['styles'].append(key)
        for key, value in colors.items():
            if value in txt:
                style['c'] = key
        for key, value in styles_dict.items():
            if value in txt:
                style['bg'] = key
        return style

    raw_txt = '\033['
    txt += '\033[0;0;0m'
    if isinstance(styles, str):
        styles = [styles]

    if styles is not None:
        for style in styles:
            raw_txt += styles_dict[style]

    if c is not None:
        if c in ['rand', 'random']:
            c = randint(0, len(colors) - 1)
        if isinstance(c, int):
            c = list(colors.keys())[c]
        raw_txt += colors[c]

    if bg is not None:
        if bg in ['rand', 'random']:
            bg = randint(0, len(backgrounds))
        if isinstance(bg, int):
            bg = list(backgrounds.keys())[bg]
        raw_txt += backgrounds[bg]

    if (c is None) and (bg is None) and (styles is None):
        raw_txt = txt
    else:
        raw_txt = raw_txt[:-1] + 'm'
        raw_txt += txt
    if p:
        print(raw_txt)
    return raw_txt


def split(txt, *delimiters):
    if len(delimiters) == 0:
        delimiters = [' ']
    elif isinstance(delimiters, str) or isinstance(delimiters, int):
        delimiters = [delimiters]

    txt = txt.split(delimiters[0])
    for d in delimiters[1:]:
        temp = []
        for w in txt:
            temp += w.split(d)
        txt = temp

    if isinstance(txt, str):
        txt = [txt]
    return txt


def replace(text, chars, to=''):
    if isinstance(chars, str):
        chars = [chars]
    for c in chars:
        text = text.replace(c, to)
    return text


def google(txt, num=5, start=1, show=False):
    sites = {'wikipedia': 'wikipedia'
        , 'imdb': 'imdb'
        , 'github': 'github'
        , 'mathoverflow': 'math overflow'
        , 'stackexchange': 'stack exchange'}
    results = []
    url = 'https://www.googleapis.com/customsearch/v1'
    payload = {'key': 'AIzaSyAx9XcWd1AwIeCz-LcNzoPE9my8pzzV9bw',
               'cx': '57c65453bf08448b1',
               'q': txt,
               'num': num,
               'start': start}

    url += '?' + '&'.join([f'{key}={value}' for key, value in payload.items()])
    search = requests.get(url).json()
    search = search['items']

    for result in search:
        r = {'link': result['link'],
             'title': result['title'],
             'synopsys': result['snippet'],
             'site': None}
        for key, site in sites.items():
            if key in r['link']:
                r['site'] = site
        results.append(r)

    if show:
        for result in results:
            print(
                f'{cprint(result["title"], c="r", styles=["b", "u"])}\n{cprint("Link", c="g", styles="u")}: {result["link"]}')
            print(wrap_text(f'{cprint("Synopsys", c="g", styles="u")}: {result["synopsys"]}'), '\n' + '-' * 50 + '\n')

    return results


def get_now(tformat='%Y_%m_%d__%H_%M_%S'):
    from datetime import datetime
    return datetime.now().strftime(tformat)


def wrap_text(txt, line=50):
    i = line
    while True:
        try:
            if txt[i - 1] != ' ':
                if txt[i - 2] != ' ':
                    txt = txt[:i - 1] + '-' + txt[i - 1:]
                else:
                    txt = txt[:i - 1] + ' ' + txt[i - 1:]
            txt = txt[:i] + '\n' + txt[i:]
            i += line
        except IndexError:
            break
    return txt


def download(url, file, data=None, auth=None, headers=None, overwrite=False, unzip=True, verify=False, session=None):
    send_headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'}
    if headers is not None:
        send_headers = send_headers | headers
    if session is None:
        r = requests.get(url, allow_redirects=True, verify=verify, data=data, headers=send_headers)
    else:
        r = session.get(url, allow_redirects=True, verify=verify, data=data, headers=send_headers)
    content_type = r.headers.get('content-type').lower()
    if os.path.isdir(file):
        name = url.split('/')[-1]
        name = name.split('?')[0]
        name = name.split('#')[0]
        file = os.path.join(file, name)

    if not overwrite:
        copy_file(file, add_time_stamp=False)
    open(file, 'wb').write(r.content)

    if (os.path.splitext(file)[-1] == '.gz') and unzip:
        import gzip
        import gzinfo
        lib = gzip.open(file, 'rb')
        file = os.path.join(os.path.dirname(file), gzinfo.read_gz_info(file).fname)
        if not overwrite:
            copy_file(file, add_time_stamp=False)
        with open(file, 'wb') as f_out:
            shutil.copyfileobj(lib, f_out)

    return file


def format_text(txt, form='file'):
    prepositions = ['Of', 'In', 'The', 'For', 'On', 'At', 'As', 'By']
    exp_letters = [' ', '(', ')', ',', '.', '\\', '/', '!', '@', '#', '$', '%', '^', '&', '*', '+', '=', '`', '~']

    match form:
        case 'title':
            txt = txt.replace('_', ' ')
            txt = txt.title()
            for word in prepositions:
                if word in txt:
                    txt = txt.replace(f' {word} ', f' {word.lower()} ')
        case 'file':
            txt = txt.lower()
            for letter in exp_letters:
                if letter in txt:
                    txt = txt.replace(letter, '_')

    return txt


def OR(event1, event2):
    if (event1 is None) or (not event1) or (event1 == 0):
        return event2
    if (event2 is None) or (not event2) or (event2 == 0):
        return event1


def runAHK(path, *args):
    cmd = f'"C:\\program files\\AutoHotkey\\AutoHotkey.exe" {path}'
    if len(args) > 0:
        for arg in args:
            cmd += f' {arg}'
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    # p.communicate('abcd')
    out = p.stdout.read()
    # print(x)
    p.stdout.close()
    return out


# def text_to_latex(text, pylatexenc=None):
#     from pylatexenc.latexencode import unicode_to_latex
#     latex = unicode_to_latex(text)
#     result = re.findall(r'\\ensuremath{.*}', latex)
#     for s in result:
#         latex.replace('\\ensuremath{%s}' % s, s)
#     return latex


def pop(lst, i1, i2, insert=None):
    x = lst[i1:i2]
    del lst[i1:i2]
    if insert is not None:
        lst.insert(i1, insert)
    return lst, x


def regedit(action, path, value=None, content='', T='REG_SZ'):
    import winreg
    sub_key = None
    base_dict = {
        'HKEY_CURRENT_USER': winreg.HKEY_CURRENT_USER,
        'HKCU': winreg.HKEY_CURRENT_USER,
        'HKCR': winreg.HKEY_CURRENT_ROOT,
        'HKEY_CURRENT_ROOT': winreg.HKEY_CURRENT_ROOT,
        'HKLM': winreg.HKEY_LOCAL_MACHINE,
        'HKEY_LOCAL_MACHINE': winreg.HKEY_LOCAL_MACHINE,
        'HKEY_USERS': winreg.HKEY_USERS,
        'HKCC': winreg.HKEY_CURRENT_CONFIG,
        'HKEY_CURRENT_CONFIG': winreg.HKEY_CURRENT_CONFIG
    }
    word_dict = {
        'REG_SZ': winreg.REG_SZ,
        'REG_BINARY': winreg.REG_BINARY,
        'REG_DWORD': winreg.REG_DWORD,
        'REG_QWORD': winreg.REG_QWORD,
        'REG_NONE': winreg.REG_NONE,
        'REG_LINK': winreg.REG_LINK,
        'REG_MULTI_SZ': winreg.REG_MULTI_SZ,
        'REG_EXPAND_SZ': winreg.REG_EXPAND_SZ
    }

    if action == 'enum':
        value = 'enum'

    path_split = split(path, '\\', '/')
    if path_split[0] in ['computer', 'Computer']:
        path_split = path_split[1:]
    base = base_dict[path_split[0]]
    if value is None:
        sub_key = path_split[-1]
        path_split = path_split[1:-1]

    path = '\\'.join(path_split[1:])
    with winreg.CreateKeyEx(base, path, access=winreg.KEY_ALL_ACCESS) as key:
        if action == 'enum':
            sub_keys = []
            values = []
            i = -1

            n_key, n_val, _ = winreg.QueryInfoKey(key)
            lim = max(n_key, n_val)
            while True:
                i += 1
                if i < n_key:
                    sub_keys.append(winreg.EnumKey(key, i))
                if i < n_val:
                    values.append(winreg.EnumValue(key, i))
                if i >= lim:
                    break
            return sub_keys, values

        if action == 'delete':
            if sub_key is not None:
                winreg.DeleteValue(key, sub_key)
            else:
                winreg.DeleteKeyEx(key, value)
            return

        if action in ['read', 'append']:
            if sub_key is not None:
                result = winreg.QueryValue(key, sub_key)
            else:
                result = winreg.QueryValueEx(key, value)
            result = result[0]

            if action == 'append':
                content = result + content
            else:
                return result

        if action in ['write', 'append']:
            T = word_dict[T]
            if sub_key is not None:
                winreg.SetValue(key, sub_key, T, content)
            else:
                winreg.SetValueEx(key, value, 0, T, content)
            return content


class RainMeter:
    NIS = '\u20AA'
    regex_element = r'(?s)<$tag$>(.*?)</$tag$>'
    colors = {'red': '255,0,0,170',
              'green': '0,255,0,200',
              'blue': '0,0,255,200'}

    def __init__(self, name, title=None, folder=None, template=None, bg=None):
        self.name = name
        self.title = title
        self.background = bg
        self.update = 1000
        self.template = template
        self.folder = folder
        self.measures = []
        self.meters = []
        self.regex = 'rexp=(?siU)'
        self.formatted_data = ''
        self.names = []
        self.objects = []
        self.variables = {}

    def append(self, name, *values, title=None, index=1, minmax=None, X=None, Y=None, y=None):
        if X is None:
            X = [10, 200]
        if Y is None:
            Y = '30r'
        if y is None:
            y = '20r'
        if title is not None:
            title = title.title()
        name = name.title()
        name = name.replace(' ', '')
        name = name.replace('-', '')
        name = name.replace('&', '')
        if not isinstance(values, tuple):
            values = [values]
        else:
            values = list(values)

        measure = [f'[Measure{name}]\n',
                   'Measure=WebParser\n',
                   'URL=[MeasureData]\n',
                   'StringIndex=%d\n' % index,
                   '\n'
                   ]
        if minmax is not None:
            self.variables[f'{name}Max'] = str(minmax[1])
            val = f'%1/#{name}Max#'
            measure = measure[:-1] + [f'MinValue={minmax[0]}\n',
                                      f'MaxValue={minmax[1]}\n',
                                      '\n',
                                      f'[MeasureCalc{name}]\n',
                                      'Measure=Calc\n',
                                      f'formula=#{name}Max#-Measure{name}\n',
                                      '\n'
                                      ]
        else:
            val = '%1#shekel#'
        if len(values) > 1:
            measure += [f'[Measure{name}Avg]\n',
                        'Measure=WebParser\n',
                        'URL=[MeasureData]\n',
                        f'StringIndex={index + 1}\n',
                        '\n'
                        ]

        meter = []
        if title is not None:
            meter += ['[MeterLabel%s]\n' % name,
                      'Meter=String\n',
                      'MeterStyle=styleLeftText\n'
                      f"FontColor={self.colors['green']}\n",
                      'StringStyle=Bold\n',
                      'InlineSetting=Underline\n',
                      'Y=%s\n' % Y,
                      'Text=%s:\n' % title,
                      '\n']
        meter += ['[meterValue%s]\n' % name,
                  'Meter=String\n',
                  'MeterStyle=styleRightText\n',
                  'MeasureName=Measure%s\n' % name,
                  f"FontColor={self.colors['green']}\n",
                  'Y=0r\n',
                  'NumOfDecimals=1\n',
                  'Text=%s\n' % val,
                  '\n'
                  ]
        if minmax is not None:
            meter = meter + ['[meterBar%s]\n' % name,
                             'Meter=Bar\n',
                             'MeterStyle=styleBar\n',
                             'MeasureName=Measure%s\n' % name,
                             'StringStyle=Bold\n',
                             'Y=%s\n' % y,
                             '\n',
                             '[meterLeft%s]\n' % name,
                             'Meter=String\n',
                             'MeterStyle=styleRightText\n',
                             f"FontColor={self.colors['red']}\n",
                             f'MeasureName=MeasureCalc{name}\n',
                             'Y=0r\n',
                             'Text=Left:%1#shekel#\n',
                             '\n',
                             ]
            if len(values) > 1:
                meter += ['[meterAvg%s]\n' % name,
                          'Meter=String\n',
                          'MeterStyle=styleLeftText\n',
                          'MeasureName=Measure%sAvg\n' % name,
                          f"FontColor={self.colors['red']}\n",
                          'Y=0r\n',
                          'NumOfDecimals=1\n',
                          f'Text=AVG:<{values[1]}#shekel#>\n',
                          '\n'
                          ]

        self.objects.append({'name': name, 'measure': measure, 'meter': meter, 'data': values})

    def build(self):
        regex = []
        for obj in self.objects:
            i = 0
            for data in obj['data']:
                i += 1
                if isinstance(data, int) or (isinstance(data, str) and data.isnumeric()):
                    sep = r'(\d*)'
                else:
                    sep = '(.*)'
                self.formatted_data += f"<{obj['name']}{i}>{data}</{obj['name']}{i}>\n"
                regex.append(f"<{obj['name']}{i}>{sep}</{obj['name']}{i}>")
            self.measures += obj['measure']
            self.meters += obj['meter']
        self.regex += '.*'.join(regex) + '\n'

    def save(self, change_ini=False):
        self.build()
        file_name = self.name.lower().replace(' ', '_')
        template = open(self.template, 'r').readlines()

        i = template.index(';$$$/variables$$$\n')
        for key, val in self.variables.items():
            template.insert(i, f'{key}={val}\n')
        i = template.index(';$$$path$$$\n') + 1
        template[i] = f'datapath=file://#CURRENTPATH#/{file_name}.txt\n'
        i = template.index(';$$$regex$$$\n') + 1
        template[i] = self.regex
        i = template.index(';$$$/title$$$\n') - 1
        template[i] = f'Text={self.name.title()}\n'

        i = template.index(';$$$measures$$$\n') + 1
        final = template[:i] + self.measures + template[i:] + self.meters

        if change_ini:
            with open(os.path.join(self.folder, '{}.ini'.format(file_name)), mode='w', encoding='utf-8') as file:
                file.truncate(0)
                file.writelines(final)

        with open(os.path.join(self.folder, '{}.txt'.format(file_name)), mode='w', encoding='utf-8') as data:
            data.truncate(0)
            data.write(self.formatted_data)

    @classmethod
    def search_tag(cls, txt, tag):
        return re.findall(cls.regex_element.replace('$tag$', tag), txt)


class ClassTemplate(object):
    def __new__(cls, name=''):
        cls.name = name

    def __init__(self, path=None, ):
        if path is not None:
            self.load(path)
        else:
            pass

    def save(self):
        file_name = save(self.__dict__, Path(self.name))
        print('File saved: "{}"'.format(file_name))

    def load(self, name):
        path = Path(name)
        attributes = load(path)
        self.__dict__.update(attributes)
        return self


class Note:
    def __init__(self, note, octave=1, length=1, amplitude=1, sample_rate=44100, cutoff=200):
        self.note = note
        self.octave = octave
        self.length = length
        self.rate = sample_rate
        self.amplitude = amplitude
        self.cutoff = cutoff
        self.wave = 0
        self.t = 0

        self.make_wave()

    def __add__(self, other):
        if isinstance(self.note, list):
            la = self.note
        else:
            la = [self.note]
        if isinstance(other, list):
            lb = other
        else:
            lb = [other]
        self.note = la + lb
        self.wave += other.wave

    def plot(self, color='black'):
        colors = ['black', 'blue', 'red', 'green', 'yellow', 'pink', 'magenta', 'cyan']
        if isinstance(color, int):
            color = colors[color]

        plt.ion()
        plt.show()
        plt.axis([0, self.cutoff, -5, 5])
        plt.plot(self.t[:self.cutoff], self.wave[:self.cutoff], color=color)
        plt.draw()
        plt.pause(0.1)

    def play(self, stream=None):
        import pyaudio
        if stream is None:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=True)

        stream.write(self.wave.astype(np.float32).tostring())

    def make_wave(self):
        if isinstance(self.note, list):
            for n in self.note:
                w, t = self.make_pitch(n, self.octave, self.length, self.rate)
                self.wave += w
                self.t = t
        else:
            self.wave, self.t = self.make_pitch(self.note, self.octave, self.length, self.rate)

        self.wave = self.amplitude * self.wave / np.max(self.wave)

    @staticmethod
    def make_pitch(note, octave, length, sample_rate=44100, shift=0):
        notes_dict = {'do': 261.63, 're': 293.66, 'mi': 329.63, 'fa': 349.23, 'sol': 392, 'la': 440, 'si': 493.88}
        if isinstance(note, str):
            note = notes_dict[note]
        frequency = note * (2 ** octave)
        length = int(length * sample_rate)
        factor = float(frequency) * (Pi * 2) / sample_rate
        waveform = np.sin((np.arange(length) + shift) * factor)
        return waveform, np.arange(length)


class Sorter:
    def __init__(self, path=None):
        self.dict = {}

        if path is not None:
            self.load(path)

    def __getitem__(self, keyword):
        # extracts item by keyword
        pass

    def __call__(self, *args, **kwargs):
        # inserts new sorting criterion
        pass

    def save(self):
        pass

    def load(self, path):
        pass