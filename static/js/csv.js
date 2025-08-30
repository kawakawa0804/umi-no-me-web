$(async function () {
  const res  = await fetch('/csv-data');
  const data = await res.json();
  const cols = Object.keys(data[0] || {}).map(k => ({ title: k, data: k }));

  new DataTable('#csvTable', {
    data, columns: cols,
    paging: true, searching: true,
    order: [[0, 'desc']],
    language: { url: '//cdn.datatables.net/plug-ins/1.13.7/i18n/ja.json' }
  });
});
